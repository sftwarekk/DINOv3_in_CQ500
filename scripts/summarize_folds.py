#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = ["macro_auc", "accuracy", "precision", "recall", "f1", "loss"]


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_row_for_fold(fold_dir: Path) -> dict:
    history_path = fold_dir / "history.json"
    summary_path = fold_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary.get("best"):
            return summary["best"]
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.json or summary.json in {fold_dir}")
    history = read_json(history_path)
    if not history:
        raise RuntimeError(f"Empty history: {history_path}")
    return max(history, key=lambda row: row.get("best_score", float("-inf")))


def flatten_fold_metrics(fold_dir: Path) -> dict:
    row = best_row_for_fold(fold_dir)
    val = row.get("val", {})
    out = {
        "fold": fold_dir.name,
        "epoch": row.get("epoch"),
        "best_score": row.get("best_score"),
    }
    for key in METRIC_KEYS:
        out[key] = val.get(key)
    return out


def summarize(rows: list[dict]) -> dict:
    mean = {}
    std = {}
    for key in METRIC_KEYS:
        values = [row[key] for row in rows if row.get(key) is not None]
        mean[key] = float(np.mean(values)) if values else None
        std[key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return {"folds": rows, "mean": mean, "std": std}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CQ-500 DINOv3 MIL fold results.")
    parser.add_argument("--run-root", required=True, type=Path, help="Directory containing fold0, fold1, ...")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", default=None, type=Path)
    args = parser.parse_args()

    fold_dirs = sorted([path for path in args.run_root.glob("fold*") if path.is_dir()])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {args.run_root}")

    rows = [flatten_fold_metrics(path) for path in fold_dirs]
    result = summarize(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    csv_path = args.output_csv or args.output_json.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(result["mean"], indent=2))
    print(f"[SAVED] {args.output_json}")
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
