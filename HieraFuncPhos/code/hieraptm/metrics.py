"""Numerical components extracted from the frozen final Task2 snapshot."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef)

def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int8)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }

def metrics(labels, probabilities) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    result = binary_metrics(labels, probabilities)
    predictions = probabilities >= 0.5
    negatives = labels == 0
    tn = int((~predictions & negatives).sum())
    fp = int((predictions & negatives).sum())
    result["specificity"] = tn / max(tn + fp, 1)
    result["rows"] = int(len(labels))
    result["positive"] = int(labels.sum())
    result["negative"] = int(len(labels) - labels.sum())
    return result


def subgroup_metrics(frame: pd.DataFrame) -> dict:
    groups = {
        "ST": frame.residue_id.isin([0, 1]),
        "Y": frame.residue_id.eq(2),
        "S": frame.residue_id.eq(0),
        "T": frame.residue_id.eq(1),
    }
    return {
        name: metrics(frame.loc[mask, "mmf_label"], frame.loc[mask, "probability"])
        for name, mask in groups.items()
    }
