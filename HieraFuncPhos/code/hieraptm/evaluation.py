"""Evaluate one selected final model, without fitting or combining models."""
from __future__ import annotations

import gc

import numpy as np
import torch

from .anchors import extract
from .config import load
from .data import DynamicFeatures, FeatureCache
from .dynamic_math import configuration, predict, tensor_sha
from .metrics import metrics, subgroup_metrics
from .models.dynamics import HieraDynLGV2
from .pipeline import contract_for
from .utils import read, save, sha


def evaluate(config_path, replicate, split="test"):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    options = load(config_path, replicate)
    torch.set_num_threads(options.dynamic.threads)
    cache = FeatureCache(options.cache, options.splits, replicate)
    contract = contract_for(options, cache)
    complete = read(options.run / "complete.json")
    if complete["contract"] != contract:
        raise RuntimeError("Run configuration, source code, weights, features, or fixed split changed")
    for stage, name, field in (("functional", "best.pt", "functional_sha256"),
                                ("dynamics", "best_adapter.pt", "dynamics_sha256")):
        if sha(options.run / stage / name) != complete[field]:
            raise RuntimeError(f"Selected {stage} checkpoint changed")
    saved = torch.load(options.run / "dynamics/best_adapter.pt", map_location="cpu", weights_only=False)
    if saved["replicate"] != replicate or saved["source_sha256"]["run_contract"] != contract:
        raise RuntimeError("Dynamics checkpoint does not belong to this replicate/run")
    if saved["config"] != configuration(options.dynamic):
        raise RuntimeError("Dynamics checkpoint configuration differs")
    device = torch.device(options.dynamic.device)
    anchors, head, anchor_source = extract(cache, options.lora, options.weights,
                                          options.run / "functional/best.pt", contract, (split,), device)
    if (anchor_source["functional_checkpoint"] != saved["source_sha256"]["functional_checkpoint"] or
            tensor_sha(head) != saved["source_sha256"]["frozen_head"]):
        raise RuntimeError("Dynamics checkpoint and frozen functional reader do not match")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    features = DynamicFeatures(cache)
    # Reapply saved TRAINING statistics; do not fit normalization on evaluation.
    features.normalize(cache.roles["train"], record=saved["normalization"])
    model = HieraDynLGV2(head, options.dynamic.dropout, options.dynamic.max_logit_delta).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    if tensor_sha(model.head.state_dict()) != tensor_sha(head):
        raise RuntimeError("Final dynamics checkpoint changed the frozen functional head")
    model.requires_grad_(False).eval()
    ids = cache.roles[split]
    probability, _ = predict(model, features, anchors, ids, options.dynamic, device)
    if not np.isfinite(probability).all():
        raise RuntimeError("Nonfinite final probabilities")
    frame = cache.index.iloc[ids][["site_id", "mmf_label", "residue", "residue_id", "mmf_split"]].copy()
    frame["probability"] = probability
    report = {"replicate": replicate, "split": split, "selected_epoch": saved["selected_epoch"],
              "checkpoint_sha256": complete["dynamics_sha256"],
              "STY": metrics(frame.mmf_label, frame.probability), "subgroups": subgroup_metrics(frame),
              "single_selected_model": True, "normalization_refitted": False}
    directory = options.run / "evaluation"
    directory.mkdir(exist_ok=True)
    frame.to_parquet(directory / f"{split}_predictions.parquet", index=False)
    save(directory / f"{split}_metrics.json", report)
    print(report, flush=True)
    return report
