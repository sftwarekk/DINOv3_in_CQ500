from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from .data import CQ500CaseDataset, build_cases, collate_cases
from .models import DINOv3MIL
from .train import get_hf_token, make_criterion, run_eval
from .utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved DINOv3 + MIL checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--split-csv", default=None, type=Path)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = dict(ckpt["args"])
    if args.split_csv is not None:
        cfg["split_csv"] = args.split_csv
    if args.project_root is not None:
        cfg["project_root"] = args.project_root
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    cfg["val_split"] = args.split
    cfg["cpu"] = args.cpu or cfg.get("cpu", False)
    ns = SimpleNamespace(**cfg)

    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not ns.cpu else "cpu")
    hf_token = get_hf_token(getattr(ns, "hf_token", None))
    image_mean = ckpt.get("image_mean", [0.485, 0.456, 0.406])
    image_std = ckpt.get("image_std", [0.229, 0.224, 0.225])

    cases, label_names = build_cases(
        ns.split_csv,
        project_root=ns.project_root,
        fold=ns.fold,
        split=args.split,
        case_id_col=ns.case_id_col,
        label_col=ns.label_col,
        label_cols=ns.label_cols,
        fold_col=ns.fold_col,
        split_col=ns.split_col,
        path_col=ns.path_col,
    )
    dataset = CQ500CaseDataset(
        cases,
        image_size=ns.image_size,
        image_mean=image_mean,
        image_std=image_std,
        max_slices=ns.max_slices,
        slice_sampling=ns.slice_sampling,
    )
    loader = DataLoader(
        dataset,
        batch_size=ns.batch_size,
        shuffle=False,
        num_workers=ns.num_workers,
        pin_memory=True,
        collate_fn=collate_cases,
        persistent_workers=ns.num_workers > 0,
    )

    model = DINOv3MIL(
        model_name=ns.model_name,
        num_labels=len(label_names),
        strategy=ns.strategy,
        pooler="mean" if ns.strategy == "linear_probe" else ns.pooler,
        feature_mode=ns.feature_mode,
        hf_token=hf_token,
        trust_remote_code=not ns.no_trust_remote_code,
        local_files_only=ns.local_files_only,
        lora_r=ns.lora_r,
        lora_alpha=ns.lora_alpha,
        lora_dropout=ns.lora_dropout,
        lora_targets=ns.lora_targets,
        unfreeze_last_n_blocks=ns.unfreeze_last_n_blocks,
        train_norm=ns.train_norm,
        train_patch_embed=ns.train_patch_embed,
        gradient_checkpointing=False,
        mil_hidden_dim=ns.mil_hidden_dim,
        mil_attn_dim=ns.mil_attn_dim,
        mil_layers=ns.mil_layers,
        mil_heads=ns.mil_heads,
        mil_dropout=ns.mil_dropout,
        transmil_max_slices=ns.transmil_max_slices,
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"[INFO] loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    criterion = make_criterion(cases, False, device)
    metrics = run_eval(
        model,
        loader,
        criterion=criterion,
        device=device,
        args=ns,
        label_names=label_names,
        epoch=int(ckpt.get("epoch", -1)),
        prediction_csv=output_dir / f"{args.split}_predictions.csv",
    )
    write_json(metrics, output_dir / f"{args.split}_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
