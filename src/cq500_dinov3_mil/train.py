from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CQ500CaseDataset, CQ500FeatureDataset, build_cases, collate_cases, collate_features
from .metrics import multilabel_metrics
from .models import DINOv3MIL
from .utils import count_parameters, ensure_dir, get_hf_token, set_seed, trainable_state_dict, write_json


def parse_float_list(value: str | None, fallback: list[float]) -> list[float]:
    if value is None:
        return fallback
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def resolve_image_stats(args: argparse.Namespace, hf_token: str | None) -> tuple[list[float], list[float]]:
    mean = parse_float_list(args.image_mean, [0.485, 0.456, 0.406])
    std = parse_float_list(args.image_std, [0.229, 0.224, 0.225])
    if args.no_processor_stats:
        return mean, std

    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(
            args.model_name,
            token=hf_token,
            trust_remote_code=not args.no_trust_remote_code,
            local_files_only=args.local_files_only,
        )
        mean = list(getattr(processor, "image_mean", mean))
        std = list(getattr(processor, "image_std", std))
    except Exception as exc:
        print(f"[WARN] could not load AutoImageProcessor stats, using provided/default stats: {exc}")
    return mean, std


def get_amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def make_loaders(args: argparse.Namespace, image_mean: list[float], image_std: list[float]):
    train_cases, label_names = build_cases(
        args.split_csv,
        project_root=args.project_root,
        fold=args.fold,
        split=args.train_split,
        case_id_col=args.case_id_col,
        label_col=args.label_col,
        label_cols=args.label_cols,
        fold_col=args.fold_col,
        split_col=args.split_col,
        path_col=args.path_col,
    )
    val_cases, _ = build_cases(
        args.split_csv,
        project_root=args.project_root,
        fold=args.fold,
        split=args.val_split,
        case_id_col=args.case_id_col,
        label_col=args.label_col,
        label_cols=args.label_cols,
        fold_col=args.fold_col,
        split_col=args.split_col,
        path_col=args.path_col,
    )

    train_ds = CQ500CaseDataset(
        train_cases,
        image_size=args.image_size,
        image_mean=image_mean,
        image_std=image_std,
        max_slices=args.max_slices,
        slice_sampling=args.slice_sampling,
    )
    val_ds = CQ500CaseDataset(
        val_cases,
        image_size=args.image_size,
        image_mean=image_mean,
        image_std=image_std,
        max_slices=args.max_slices,
        slice_sampling=args.slice_sampling,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate_cases,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_cases, val_cases, label_names


def make_model(args: argparse.Namespace, num_labels: int, hf_token: str | None) -> DINOv3MIL:
    pooler = "mean" if args.strategy == "linear_probe" else args.pooler
    return DINOv3MIL(
        model_name=args.model_name,
        num_labels=num_labels,
        strategy=args.strategy,
        pooler=pooler,
        feature_mode=args.feature_mode,
        hf_token=hf_token,
        trust_remote_code=not args.no_trust_remote_code,
        local_files_only=args.local_files_only,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        train_norm=args.train_norm,
        train_patch_embed=args.train_patch_embed,
        gradient_checkpointing=args.gradient_checkpointing,
        mil_hidden_dim=args.mil_hidden_dim,
        mil_attn_dim=args.mil_attn_dim,
        mil_layers=args.mil_layers,
        mil_heads=args.mil_heads,
        mil_dropout=args.mil_dropout,
        transmil_max_slices=args.transmil_max_slices,
    )


def make_criterion(train_cases, use_pos_weight: bool, device: torch.device) -> nn.Module:
    if not use_pos_weight:
        return nn.BCEWithLogitsLoss()
    labels = np.stack([case.labels for case in train_cases], axis=0).astype(np.float32)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    weights = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    print(f"[INFO] pos_weight={weights.tolist()}")
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32, device=device))


def optimizer_groups(args: argparse.Namespace, model: DINOv3MIL) -> list[dict[str, Any]]:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr_backbone, "weight_decay": args.weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay})
    if not groups:
        raise RuntimeError("No trainable parameters found.")
    return groups


def run_eval(
    model: DINOv3MIL,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace | SimpleNamespace,
    label_names: list[str],
    epoch: int,
    prediction_csv: Path | None = None,
) -> dict:
    model.eval()
    logits_all = []
    labels_all = []
    rows = []
    loss_sum = 0.0
    count = 0
    amp_dtype = get_amp_dtype(args.amp_dtype)

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                if "features" in batch:
                    features = batch["features"].to(device, non_blocking=True)
                    out = model.classify_features(features, mask)
                    batch_size = features.shape[0]
                else:
                    images = batch["images"].to(device, non_blocking=True)
                    out = model(images, mask, slice_chunk_size=args.slice_chunk_size)
                    batch_size = images.shape[0]
                loss = criterion(out["logits"], labels)

            probs = torch.sigmoid(out["logits"]).detach().float().cpu().numpy()
            logits_np = out["logits"].detach().float().cpu().numpy()
            labels_np = labels.detach().float().cpu().numpy()
            logits_all.append(logits_np)
            labels_all.append(labels_np)
            loss_sum += float(loss.detach().cpu().item()) * batch_size
            count += batch_size

            for i, case_id in enumerate(batch["case_id"]):
                row = {"epoch": epoch, "case_id": case_id}
                for j, name in enumerate(label_names):
                    row[f"{name}_label"] = float(labels_np[i, j])
                    row[f"{name}_prob"] = float(probs[i, j])
                rows.append(row)

    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    metrics = multilabel_metrics(logits, labels, label_names=label_names, threshold=args.threshold)
    metrics["loss"] = float(loss_sum / max(count, 1))

    if prediction_csv is not None:
        ensure_dir(prediction_csv.parent)
        pd.DataFrame(rows).to_csv(prediction_csv, index=False)
    return metrics


@torch.no_grad()
def precompute_feature_items(
    model: DINOv3MIL,
    loader: DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    desc: str,
) -> list[dict]:
    model.eval()
    items = []
    for batch in tqdm(loader, desc=desc):
        images = batch["images"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        out = model(images, mask, slice_chunk_size=args.slice_chunk_size)
        features = out["features"].detach().float().cpu()
        labels = batch["labels"].detach().float().cpu()
        for i, case_id in enumerate(batch["case_id"]):
            n = int(mask[i].sum().item())
            items.append(
                {
                    "case_id": case_id,
                    "features": features[i, :n].contiguous(),
                    "labels": labels[i],
                    "slice_paths": batch["slice_paths"][i],
                }
            )
    return items


def maybe_cache_frozen_features(
    args: argparse.Namespace,
    model: DINOv3MIL,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    if not args.cache_frozen_features:
        return train_loader, val_loader
    if args.strategy not in {"frozen", "linear_probe"}:
        print("[WARN] --cache-frozen-features only applies to frozen/linear_probe; ignoring.")
        return train_loader, val_loader

    train_items = precompute_feature_items(model, train_loader, device=device, args=args, desc="cache train features")
    val_items = precompute_feature_items(model, val_loader, device=device, args=args, desc="cache val features")
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": True,
        "collate_fn": collate_features,
    }
    return (
        DataLoader(CQ500FeatureDataset(train_items), shuffle=True, **kwargs),
        DataLoader(CQ500FeatureDataset(val_items), shuffle=False, **kwargs),
    )


def save_checkpoint(
    path: Path,
    *,
    model: DINOv3MIL,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
    label_names: list[str],
    image_mean: list[float],
    image_std: list[float],
    save_full_state: bool,
) -> None:
    ensure_dir(path.parent)
    model_state = model.state_dict() if save_full_state else trainable_state_dict(model)
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model": {key: value.detach().cpu() for key, value in model_state.items()},
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "label_names": label_names,
            "image_mean": image_mean,
            "image_std": image_std,
            "trainable_info": model.trainable_info,
            "param_summary": count_parameters(model),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    pred_dir = ensure_dir(output_dir / "predictions")

    hf_token = get_hf_token(args.hf_token)
    image_mean, image_std = resolve_image_stats(args, hf_token)
    train_loader, val_loader, train_cases, val_cases, label_names = make_loaders(args, image_mean, image_std)

    args.num_labels = len(label_names)
    write_json(vars(args), output_dir / "config.json")
    write_json({"image_mean": image_mean, "image_std": image_std, "labels": label_names}, output_dir / "preprocess.json")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] fold={args.fold} train_cases={len(train_cases)} val_cases={len(val_cases)} labels={label_names}")

    model = make_model(args, num_labels=len(label_names), hf_token=hf_token).to(device)
    write_json({"trainable_info": model.trainable_info, "params": count_parameters(model)}, output_dir / "model_trainable.json")
    print(f"[INFO] trainable={count_parameters(model)}")
    print(f"[INFO] trainable_info={model.trainable_info}")

    train_loader, val_loader = maybe_cache_frozen_features(args, model, train_loader, val_loader, device)

    optimizer = torch.optim.AdamW(optimizer_groups(args, model))
    criterion = make_criterion(train_cases, args.use_pos_weight, device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16" and device.type == "cuda")
    amp_dtype = get_amp_dtype(args.amp_dtype)

    start_epoch = 0
    best_score = -math.inf
    if args.resume:
        resume_path = ckpt_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device)
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_score = float(ckpt.get("best_score", -math.inf))
            print(f"[INFO] resumed {resume_path}; missing={len(missing)} unexpected={len(unexpected)}")

    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = []
        train_labels = []
        train_loss_sum = 0.0
        train_count = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}"), start=1):
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                if "features" in batch:
                    features = batch["features"].to(device, non_blocking=True)
                    out = model.classify_features(features, mask)
                    batch_size = features.shape[0]
                else:
                    images = batch["images"].to(device, non_blocking=True)
                    out = model(images, mask, slice_chunk_size=args.slice_chunk_size)
                    batch_size = images.shape[0]
                loss = criterion(out["logits"], labels)
                backward_loss = loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            train_loss_sum += float(loss.detach().cpu().item()) * batch_size
            train_count += batch_size
            train_logits.append(out["logits"].detach().float().cpu().numpy())
            train_labels.append(labels.detach().float().cpu().numpy())

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                if args.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        train_metrics = multilabel_metrics(
            np.concatenate(train_logits, axis=0),
            np.concatenate(train_labels, axis=0),
            label_names=label_names,
            threshold=args.threshold,
        )
        train_metrics["loss"] = float(train_loss_sum / max(train_count, 1))

        val_metrics = run_eval(
            model,
            val_loader,
            criterion=criterion,
            device=device,
            args=args,
            label_names=label_names,
            epoch=epoch,
            prediction_csv=pred_dir / f"val_epoch{epoch:03d}.csv",
        )

        monitor_value = val_metrics.get(args.monitor_metric)
        if monitor_value is None:
            monitor_value = -val_metrics["loss"]
        score = float(monitor_value)
        is_best = score > best_score
        if is_best:
            best_score = score

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "best_score": best_score, "is_best": is_best}
        history.append(row)
        write_json(history, output_dir / "history.json")
        write_json(row, output_dir / "latest_epoch.json")
        save_checkpoint(
            ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_score=best_score,
            args=args,
            label_names=label_names,
            image_mean=image_mean,
            image_std=image_std,
            save_full_state=args.save_full_state,
        )
        if is_best:
            save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_score,
                args=args,
                label_names=label_names,
                image_mean=image_mean,
                image_std=image_std,
                save_full_state=args.save_full_state,
            )
            run_eval(
                model,
                val_loader,
                criterion=criterion,
                device=device,
                args=args,
                label_names=label_names,
                epoch=epoch,
                prediction_csv=pred_dir / "val_best.csv",
            )

        print(
            f"[EPOCH {epoch}] train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_auc={val_metrics.get('macro_auc')} "
            f"val_f1={val_metrics.get('f1')} best={best_score:.6f}"
        )

    write_json({"best": max(history, key=lambda item: item["best_score"]) if history else None}, output_dir / "summary.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DINOv3 + MIL on CQ-500.")

    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--case-id-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--label-cols", default=None)
    parser.add_argument("--fold-col", default=None)
    parser.add_argument("--split-col", default=None)
    parser.add_argument("--path-col", default=None)

    parser.add_argument("--model-name", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--feature-mode", default="cls_patch_mean", choices=["cls", "cls_patch_mean"])
    parser.add_argument("--strategy", required=True, choices=["frozen", "linear_probe", "lora", "partial"])
    parser.add_argument("--pooler", default="abmil", choices=["mean", "abmil", "transmil"])

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--image-mean", default=None, help="Comma-separated RGB mean. Defaults to processor or ImageNet.")
    parser.add_argument("--image-std", default=None, help="Comma-separated RGB std. Defaults to processor or ImageNet.")
    parser.add_argument("--no-processor-stats", action="store_true")
    parser.add_argument("--max-slices", type=int, default=0, help="0 means use all slices.")
    parser.add_argument("--slice-sampling", default="uniform", choices=["uniform", "center_uniform"])

    parser.add_argument("--mil-hidden-dim", type=int, default=512)
    parser.add_argument("--mil-attn-dim", type=int, default=256)
    parser.add_argument("--mil-layers", type=int, default=2)
    parser.add_argument("--mil-heads", type=int, default=8)
    parser.add_argument("--mil-dropout", type=float, default=0.1)
    parser.add_argument("--transmil-max-slices", type=int, default=256)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-targets", default="auto")

    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=1)
    parser.add_argument("--train-norm", action="store_true")
    parser.add_argument("--train-patch-embed", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--slice-chunk-size", type=int, default=4)
    parser.add_argument("--cache-frozen-features", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--monitor-metric", default="macro_auc", choices=["macro_auc", "f1", "recall", "accuracy", "loss"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="auto or checkpoint path")
    parser.add_argument("--save-full-state", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.max_slices <= 0:
        args.max_slices = None
    train(args)


if __name__ == "__main__":
    main()
