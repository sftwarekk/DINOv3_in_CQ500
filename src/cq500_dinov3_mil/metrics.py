from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
except Exception:  # pragma: no cover
    accuracy_score = f1_score = precision_score = recall_score = roc_auc_score = None


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> dict:
    y_true = np.asarray(labels).astype(int)
    y_score = sigmoid_np(np.asarray(logits))
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_score.ndim == 1:
        y_score = y_score[:, None]

    y_pred = (y_score >= threshold).astype(int)
    names = list(label_names or [f"label_{i}" for i in range(y_true.shape[1])])

    out: dict = {
        "threshold": float(threshold),
        "num_cases": int(y_true.shape[0]),
    }

    per_label = {}
    aucs = []
    for i, name in enumerate(names):
        item = {
            "positive_cases": int(y_true[:, i].sum()),
            "negative_cases": int((1 - y_true[:, i]).sum()),
        }
        if roc_auc_score is not None and len(np.unique(y_true[:, i])) == 2:
            item["auc"] = float(roc_auc_score(y_true[:, i], y_score[:, i]))
            aucs.append(item["auc"])
        else:
            item["auc"] = None
        per_label[name] = item

    out["per_label"] = per_label
    out["macro_auc"] = float(np.mean(aucs)) if aucs else None

    if accuracy_score is not None:
        average = "binary" if y_true.shape[1] == 1 else "macro"
        yt = y_true.ravel() if y_true.shape[1] == 1 else y_true
        yp = y_pred.ravel() if y_true.shape[1] == 1 else y_pred
        out["accuracy"] = float(accuracy_score(yt, yp))
        out["precision"] = float(precision_score(yt, yp, average=average, zero_division=0))
        out["recall"] = float(recall_score(yt, yp, average=average, zero_division=0))
        out["f1"] = float(f1_score(yt, yp, average=average, zero_division=0))

    return out
