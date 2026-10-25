"""Final Task1 training dependencies, extracted without changing function bodies."""

from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be equal-length one-dimensional arrays")
    if not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Metrics require finite scores and both binary classes")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "rows": int(labels.size),
        "positives": int(labels.sum()),
        "negatives": int(labels.size - labels.sum()),
    }


def write_curves(labels: np.ndarray, scores: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, roc_threshold = roc_curve(labels, scores)
    precision, recall, pr_threshold = precision_recall_curve(labels, scores)
    _atomic_rows(
        (
            {"point": index, "fpr": x, "tpr": y, "threshold": threshold}
            for index, (x, y, threshold) in enumerate(zip(fpr, tpr, roc_threshold))
        ),
        output_dir / "roc_curve.tsv",
        ("point", "fpr", "tpr", "threshold"),
    )
    # sklearn returns one fewer PR threshold than precision/recall points.
    padded_pr_threshold = np.append(pr_threshold, np.nan)
    _atomic_rows(
        (
            {"point": index, "recall": x, "precision": y, "threshold": threshold}
            for index, (x, y, threshold) in enumerate(zip(recall, precision, padded_pr_threshold))
        ),
        output_dir / "pr_curve.tsv",
        ("point", "recall", "precision", "threshold"),
    )


def _atomic_rows(rows: Iterable[Mapping[str, Any]], path: Path, fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
