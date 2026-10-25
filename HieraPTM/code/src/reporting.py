"""Prediction tables and train/test figures."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import utils as shared
from dataset import FinalSplitData as RelaxedHeadData


def prediction_rows(data: RelaxedHeadData, selected: np.ndarray, labels: np.ndarray,
                    scores: np.ndarray, model: str, extra: dict[str, np.ndarray] | None = None) -> list[dict[str, Any]]:
    rows = []
    for position, (index, label, score) in enumerate(zip(selected, labels, scores)):
        meta = data.metadata[int(index)]
        row = {
            "sample_id": meta["sample_id"], "protein_id": meta["protein_id"],
            "center_aa": meta["center_aa"], "center_position_1based": meta["center_position_1based"],
            "ptm": meta["ptm"], "head": meta["head"], "label": int(label),
            "sequence_61": meta["sequence_61"], "split": meta["split"],
            "outer_source_split": meta["outer_source_split"], "outer_source_row": meta["outer_source_row"],
            "model": model, "model_score": f"{float(score):.12g}",
        }
        if extra:
            row.update({key: values[position] for key, values in extra.items()})
        rows.append(row)
    return rows


def plot_training(history: list[dict[str, Any]], path: Path, title: str) -> None:
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["auroc"]) for row in history], label="AUROC")
    axes[0].plot(epochs, [float(row["auprc"]) for row in history], label="AUPRC")
    axes[0].set(xlabel="Epoch", ylabel="Score", title="Development metrics"); axes[0].legend()
    axes[1].plot(epochs, [float(row["train_loss"]) for row in history], label="Train")
    axes[1].plot(epochs, [float(row["validation_loss"]) for row in history], label="Development")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Loss"); axes[1].legend()
    fig.suptitle(title); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220); plt.close(fig)


def save_test_figure(output: Path, head: str, final: dict[str, Any]) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, score in (("Sequence Backbone", final["parent_scores"]), ("HieraPTM", final["scores"])):
        fpr, tpr, _ = roc_curve(final["labels"], score)
        precision, recall, _ = precision_recall_curve(final["labels"], score)
        metrics = shared.binary_metrics(final["labels"], score)
        axes[0].plot(fpr, tpr, label=f"{label} ({metrics['auroc']:.4f})")
        axes[1].plot(recall, precision, label=f"{label} ({metrics['auprc']:.4f})")
    axes[0].plot([0, 1], [0, 1], color="0.75", lw=1); axes[0].set(xlabel="1-specificity", ylabel="Sensitivity", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR")
    for axis in axes: axis.legend()
    fig.suptitle(f"{head}: relaxed row-stratified test"); fig.tight_layout()
    fig.savefig(output / "relaxed_test_roc_pr.png", dpi=300); plt.close(fig)
