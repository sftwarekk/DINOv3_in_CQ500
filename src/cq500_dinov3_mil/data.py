from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .utils import natural_key


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


CASE_ID_ALIASES = ["name", "case_id", "case", "patient_id", "study_id", "series_id"]
LABEL_ALIASES = ["label", "target", "y", "hemorrhage", "ich", "any_hemorrhage"]
PATH_ALIASES = ["series_dir", "case_dir", "dir_path", "image_dir", "path", "local_path"]
FOLD_ALIASES = ["fold", "Fold", "cv_fold", "kfold"]
SPLIT_ALIASES = ["split", "Split", "set", "phase", "partition"]


@dataclass(frozen=True)
class CaseItem:
    case_id: str
    labels: np.ndarray
    slice_paths: list[str]


def find_column(df: pd.DataFrame, aliases: Sequence[str], user_value: str | None = None) -> str:
    if user_value:
        if user_value not in df.columns:
            raise KeyError(f"Column '{user_value}' not found. Available columns: {list(df.columns)}")
        return user_value

    lower = {str(col).lower(): str(col) for col in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise KeyError(f"Could not find any of {aliases}. Available columns: {list(df.columns)}")


def parse_label_cols(df: pd.DataFrame, label_cols: str | Sequence[str] | None, label_col: str | None) -> list[str]:
    if label_cols:
        cols = [c.strip() for c in (label_cols.split(",") if isinstance(label_cols, str) else label_cols) if c.strip()]
    elif label_col:
        cols = [label_col]
    else:
        cols = [find_column(df, LABEL_ALIASES)]

    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Label columns not found: {missing}")
    return cols


def split_matches(value: object, requested: str) -> bool:
    value_norm = str(value).strip().lower()
    requested_norm = requested.strip().lower()
    aliases = {
        "train": {"train", "tr", "training"},
        "val": {"val", "valid", "validation", "dev", "eval"},
        "test": {"test", "te"},
    }
    return value_norm in aliases.get(requested_norm, {requested_norm})


def resolve_series_dir(project_root: str | Path | None, raw_path: object) -> Path:
    raw = Path(str(raw_path))
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)

    if project_root is not None:
        root = Path(project_root)
        candidates.append(root / raw)
        parts = list(raw.parts)
        if parts and parts[0].lower() in {"datase", "dataset"} and len(parts) > 1:
            candidates.append(root / Path(*parts[1:]))

    candidates.append(raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def list_image_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    if not path.exists():
        return []

    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: natural_key(p.name))


def sample_paths(paths: Sequence[str], max_slices: int | None, strategy: str) -> list[str]:
    if max_slices is None or max_slices <= 0 or len(paths) <= max_slices:
        return list(paths)
    if strategy == "uniform":
        idx = np.linspace(0, len(paths) - 1, max_slices).round().astype(int)
    elif strategy == "center_uniform":
        start = int(0.1 * len(paths))
        end = max(start, int(0.9 * len(paths)) - 1)
        idx = np.linspace(start, end, max_slices).round().astype(int)
    else:
        raise ValueError(f"Unknown slice sampling strategy: {strategy}")
    return [paths[int(i)] for i in idx]


def build_cases(
    split_csv: str | Path,
    *,
    project_root: str | Path | None,
    fold: int,
    split: str,
    case_id_col: str | None = None,
    label_col: str | None = None,
    label_cols: str | Sequence[str] | None = None,
    fold_col: str | None = None,
    split_col: str | None = None,
    path_col: str | None = None,
) -> tuple[list[CaseItem], list[str]]:
    df = pd.read_csv(split_csv)
    case_id_col = find_column(df, CASE_ID_ALIASES, case_id_col)
    fold_col = find_column(df, FOLD_ALIASES, fold_col)
    split_col = find_column(df, SPLIT_ALIASES, split_col)
    path_col = find_column(df, PATH_ALIASES, path_col)
    labels = parse_label_cols(df, label_cols=label_cols, label_col=label_col)

    df = df[df[fold_col].astype(int) == int(fold)].copy()
    df = df[df[split_col].map(lambda x: split_matches(x, split))].copy()
    if df.empty:
        raise ValueError(f"No cases found for fold={fold}, split={split}")

    cases: list[CaseItem] = []
    failures = []
    for case_id, group in df.groupby(case_id_col, sort=False):
        label_values = group[labels].iloc[0].astype(np.float32).to_numpy()
        slice_paths: list[Path] = []
        for raw in group[path_col].dropna().unique().tolist():
            series_dir = resolve_series_dir(project_root, raw)
            slice_paths.extend(list_image_files(series_dir))
        slice_paths = sorted(set(slice_paths), key=lambda p: natural_key(str(p)))
        if not slice_paths:
            failures.append(str(case_id))
            continue
        cases.append(CaseItem(str(case_id), label_values, [str(p) for p in slice_paths]))

    if not cases:
        raise FileNotFoundError(f"No readable image cases for fold={fold}, split={split}. Failures: {failures[:10]}")
    return cases, labels


class CQ500CaseDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[CaseItem],
        *,
        image_size: int,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        max_slices: int | None = None,
        slice_sampling: str = "uniform",
    ) -> None:
        self.cases = list(cases)
        self.image_size = int(image_size)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.max_slices = max_slices
        self.slice_sampling = slice_sampling

    def __len__(self) -> int:
        return len(self.cases)

    def _load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("L")
        array = np.asarray(image, dtype=np.float32)
        if array.max() > 1.0:
            array = array / 255.0
        x = torch.from_numpy(np.clip(array, 0.0, 1.0)).view(1, 1, *array.shape)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = x.squeeze(0).repeat(3, 1, 1)
        return (x - self.image_mean) / self.image_std

    def __getitem__(self, index: int) -> dict:
        item = self.cases[index]
        paths = sample_paths(item.slice_paths, self.max_slices, self.slice_sampling)
        images = torch.stack([self._load_image(path) for path in paths], dim=0)
        return {
            "case_id": item.case_id,
            "images": images,
            "labels": torch.tensor(item.labels, dtype=torch.float32),
            "slice_paths": paths,
        }


def collate_cases(batch: list[dict]) -> dict:
    max_slices = max(item["images"].shape[0] for item in batch)
    channels, height, width = batch[0]["images"].shape[1:]
    images = torch.zeros(len(batch), max_slices, channels, height, width, dtype=batch[0]["images"].dtype)
    mask = torch.zeros(len(batch), max_slices, dtype=torch.bool)
    labels = torch.stack([item["labels"] for item in batch], dim=0)

    for i, item in enumerate(batch):
        n = item["images"].shape[0]
        images[i, :n] = item["images"]
        mask[i, :n] = True

    return {
        "case_id": [item["case_id"] for item in batch],
        "images": images,
        "mask": mask,
        "labels": labels,
        "slice_paths": [item["slice_paths"] for item in batch],
    }


class CQ500FeatureDataset(Dataset):
    def __init__(self, items: Sequence[dict]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def collate_features(batch: list[dict]) -> dict:
    max_slices = max(item["features"].shape[0] for item in batch)
    feat_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_slices, feat_dim, dtype=batch[0]["features"].dtype)
    mask = torch.zeros(len(batch), max_slices, dtype=torch.bool)
    labels = torch.stack([item["labels"] for item in batch], dim=0)

    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        features[i, :n] = item["features"]
        mask[i, :n] = True

    return {
        "case_id": [item["case_id"] for item in batch],
        "features": features,
        "mask": mask,
        "labels": labels,
        "slice_paths": [item["slice_paths"] for item in batch],
    }
