from __future__ import annotations

import re
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MeanPooler(nn.Module):
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class GatedABMILPooler(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, attn_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn_v = nn.Linear(hidden_dim, attn_dim)
        self.attn_u = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.project(features)
        scores = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        return torch.sum(attention.unsqueeze(-1) * h, dim=1)


class TransMILPooler(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_slices: int = 256,
    ) -> None:
        super().__init__()
        self.max_slices = max_slices
        self.project = nn.Linear(in_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_slices + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _position(self, length: int) -> torch.Tensor:
        if length <= self.max_slices + 1:
            return self.pos_embed[:, :length]
        pos = self.pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=length, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        x = self.project(features)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(batch, 1, dtype=torch.bool, device=mask.device)
        keep_mask = torch.cat([cls_mask, mask], dim=1)
        x = x + self._position(x.shape[1]).to(dtype=x.dtype, device=x.device)
        x = self.encoder(x, src_key_padding_mask=~keep_mask)
        return self.norm(x[:, 0])


def extract_global_feature(output: Any, *, num_register_tokens: int, mode: str) -> torch.Tensor:
    hidden = None
    pooler = None
    if isinstance(output, dict):
        hidden = output.get("last_hidden_state")
        pooler = output.get("pooler_output")
    else:
        hidden = getattr(output, "last_hidden_state", None)
        pooler = getattr(output, "pooler_output", None)
        if hidden is None and isinstance(output, (tuple, list)) and output:
            hidden = output[0]

    if hidden is not None and hidden.ndim == 3:
        cls = hidden[:, 0]
        if mode == "cls_patch_mean":
            patch_start = 1 + int(num_register_tokens)
            patch = hidden[:, patch_start:].mean(dim=1)
            return torch.cat([cls, patch], dim=-1)
        if mode == "cls":
            return cls
        raise ValueError(f"Unknown feature_mode: {mode}")

    if pooler is not None:
        return pooler

    if torch.is_tensor(output):
        return output[:, 0] if output.ndim == 3 else output
    raise RuntimeError(f"Unsupported DINOv3 output type: {type(output)}")


def infer_feature_dim(config: Any, feature_mode: str) -> int:
    hidden = getattr(config, "hidden_size", None) or getattr(config, "embed_dim", None) or getattr(config, "hidden_dim", None)
    if hidden is None:
        raise RuntimeError("Could not infer DINOv3 hidden size from model config.")
    multiplier = 2 if feature_mode == "cls_patch_mean" else 1
    return int(hidden) * multiplier


def resolve_lora_targets(backbone: nn.Module, target_modules: str | Sequence[str] | None) -> str | list[str] | None:
    if target_modules is None:
        return None
    if not isinstance(target_modules, str):
        return [str(x) for x in target_modules]

    value = target_modules.strip()
    if value in {"", "none", "None"}:
        return None
    if value == "all-linear":
        return "all-linear"
    if value != "auto":
        return [x.strip() for x in value.split(",") if x.strip()]

    leaf_names = {name.split(".")[-1] for name, module in backbone.named_modules() if isinstance(module, nn.Linear)}
    if "qkv" in leaf_names:
        return ["qkv"]
    if {"query", "value"}.issubset(leaf_names):
        return ["query", "value"]
    if {"q_proj", "v_proj"}.issubset(leaf_names):
        return ["q_proj", "v_proj"]
    return "all-linear"


def detect_block_ids(backbone: nn.Module) -> tuple[str, list[int]]:
    patterns = [
        ("blocks", re.compile(r"(?:^|\.)blocks\.(\d+)\.")),
        ("encoder.layer", re.compile(r"(?:^|\.)encoder\.layer\.(\d+)\.")),
        ("model.layer", re.compile(r"(?:^|\.)model\.layer\.(\d+)\.")),
        ("layers", re.compile(r"(?:^|\.)layers\.(\d+)\.")),
    ]
    names = [name for name, _ in backbone.named_parameters()]
    best_name = ""
    best_ids: list[int] = []
    for pattern_name, pattern in patterns:
        ids = sorted({int(match.group(1)) for name in names if (match := pattern.search(name))})
        if len(ids) > len(best_ids):
            best_name, best_ids = pattern_name, ids
    if not best_ids:
        raise RuntimeError("Could not detect transformer block ids for partial fine-tuning.")
    return best_name, best_ids


def name_in_block(name: str, pattern_name: str, block_id: int) -> bool:
    if pattern_name == "blocks":
        return re.search(rf"(?:^|\.)blocks\.{block_id}\.", name) is not None
    if pattern_name == "encoder.layer":
        return re.search(rf"(?:^|\.)encoder\.layer\.{block_id}\.", name) is not None
    if pattern_name == "model.layer":
        return re.search(rf"(?:^|\.)model\.layer\.{block_id}\.", name) is not None
    if pattern_name == "layers":
        return re.search(rf"(?:^|\.)layers\.{block_id}\.", name) is not None
    return False


class DINOv3MIL(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        strategy: str,
        pooler: str,
        feature_mode: str = "cls_patch_mean",
        hf_token: str | None = None,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_targets: str = "auto",
        unfreeze_last_n_blocks: int = 1,
        train_norm: bool = False,
        train_patch_embed: bool = False,
        gradient_checkpointing: bool = False,
        mil_hidden_dim: int = 512,
        mil_attn_dim: int = 256,
        mil_layers: int = 2,
        mil_heads: int = 8,
        mil_dropout: float = 0.1,
        transmil_max_slices: int = 256,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.pooler_name = pooler
        self.feature_mode = feature_mode

        backbone = AutoModel.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if gradient_checkpointing and hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()

        if strategy == "lora":
            from peft import LoraConfig, TaskType, get_peft_model

            targets = resolve_lora_targets(backbone, lora_targets)
            config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=targets,
            )
            backbone = get_peft_model(backbone, config)
            self.lora_targets = targets
        else:
            self.lora_targets = None

        if gradient_checkpointing and hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

        self.backbone = backbone
        self.num_register_tokens = int(getattr(backbone.config, "num_register_tokens", 0))
        self.feature_dim = infer_feature_dim(backbone.config, feature_mode)
        pooled_dim = self._build_pooler(pooler, mil_hidden_dim, mil_attn_dim, mil_layers, mil_heads, mil_dropout, transmil_max_slices)
        self.classifier = nn.Linear(pooled_dim, num_labels)
        self.trainable_info = self._configure_trainable(
            strategy=strategy,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            train_norm=train_norm,
            train_patch_embed=train_patch_embed,
        )

    def _build_pooler(
        self,
        pooler: str,
        hidden_dim: int,
        attn_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        max_slices: int,
    ) -> int:
        if pooler == "mean":
            self.pooler = MeanPooler()
            return self.feature_dim
        if pooler == "abmil":
            self.pooler = GatedABMILPooler(self.feature_dim, hidden_dim, attn_dim, dropout)
            return hidden_dim
        if pooler == "transmil":
            self.pooler = TransMILPooler(self.feature_dim, hidden_dim, layers, heads, dropout, max_slices)
            return hidden_dim
        raise ValueError(f"Unknown pooler: {pooler}")

    def _configure_trainable(
        self,
        *,
        strategy: str,
        unfreeze_last_n_blocks: int,
        train_norm: bool,
        train_patch_embed: bool,
    ) -> dict:
        for param in self.backbone.parameters():
            param.requires_grad = False

        info: dict = {"strategy": strategy}
        if strategy in {"frozen", "linear_probe"}:
            pass
        elif strategy == "lora":
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            info["lora_targets"] = self.lora_targets
        elif strategy == "partial":
            pattern_name, block_ids = detect_block_ids(self.backbone)
            target_blocks = block_ids[-unfreeze_last_n_blocks:] if unfreeze_last_n_blocks > 0 else []
            for name, param in self.backbone.named_parameters():
                if any(name_in_block(name, pattern_name, block_id) for block_id in target_blocks):
                    param.requires_grad = True
                if train_norm and ("norm" in name.lower() or "layernorm" in name.lower()):
                    param.requires_grad = True
                if train_patch_embed and ("patch_embed" in name.lower() or "embeddings.patch" in name.lower()):
                    param.requires_grad = True
            info.update({"block_pattern": pattern_name, "all_block_ids": block_ids, "target_blocks": target_blocks})
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for param in self.pooler.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        return info

    def _backbone_has_trainable_params(self) -> bool:
        return any(param.requires_grad for param in self.backbone.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and not self._backbone_has_trainable_params():
            self.backbone.eval()
        return self

    def encode_flat_slices(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=pixel_values, return_dict=True)
        return extract_global_feature(output, num_register_tokens=self.num_register_tokens, mode=self.feature_mode)

    def classify_features(self, features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.pooler(features, mask)
        logits = self.classifier(pooled)
        return {"logits": logits, "features": features, "pooled": pooled}

    def forward(self, images: torch.Tensor, mask: torch.Tensor, slice_chunk_size: int = 4) -> dict[str, torch.Tensor]:
        batch, slices, channels, height, width = images.shape
        flat_images = images.reshape(batch * slices, channels, height, width)
        flat_mask = mask.reshape(batch * slices)
        valid_images = flat_images[flat_mask]
        if valid_images.numel() == 0:
            raise RuntimeError("Batch contains no valid slices.")

        feature_chunks = []
        for start in range(0, valid_images.shape[0], slice_chunk_size):
            feature_chunks.append(self.encode_flat_slices(valid_images[start : start + slice_chunk_size]))
        valid_features = torch.cat(feature_chunks, dim=0)

        features = torch.zeros(batch * slices, valid_features.shape[-1], device=images.device, dtype=valid_features.dtype)
        features[flat_mask] = valid_features
        features = features.view(batch, slices, -1)

        return self.classify_features(features, mask)
