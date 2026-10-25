"""Fixed-test evaluation of selected weights; no optimizer or training loop."""
from __future__ import annotations
import gc
import json
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils as shared
import graph_data as graph_train
from configuration import ROOT
from models import build_graph_variant
from dataset import FinalSplitData as RelaxedHeadData
from esm_cache import ESMResidueWindowCache
from data import SafeFullProteinNodeReader, RelaxedGraphCollator, indices, make_loader
from reporting import prediction_rows, save_test_figure


PAPER_AUROC = {
    "acetylation_k": 0.90, "glycosylation_n": 0.98, "methylation_k": 0.95,
    "methylation_r": 0.96, "phosphorylation_st": 0.95, "phosphorylation_y": 0.91,
    "sumoylation_k": 0.85, "ubiquitination_k": 0.87,
}


def evaluate_final(model: Any, stream: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval(); labels: list[np.ndarray] = []; parent_scores: list[np.ndarray] = []
    scores: list[np.ndarray] = []; sample_ids: list[str] = []; nodes: list[np.ndarray] = []
    outside: list[np.ndarray] = []
    with torch.inference_mode():
        for raw in stream:
            batch = graph_train.move(raw, device)
            representation = model.encode(
                batch["token_ids"], batch["attention_mask"], batch["esm_embeddings"],
                batch["esm_present_mask"], batch["node_esm"], batch["node_positions"],
                batch["node_coordinates"], batch["node_geometry"], batch["node_mask"],
            )
            classifier = model.parent.backbone.classifier
            parent_logit = classifier(representation["sequence_center_token"])[:, 0]
            final_logit = classifier(representation["center_token"])[:, 0]
            labels.append(batch["label"].float().cpu().numpy())
            parent_scores.append(torch.sigmoid(parent_logit).float().cpu().numpy())
            scores.append(torch.sigmoid(final_logit).float().cpu().numpy())
            sample_ids.extend(raw["sample_id"])
            nodes.append(raw["node_mask"].sum(1).numpy())
            outside.append((raw["node_geometry"][:, :, 7] * raw["node_mask"]).sum(1).numpy())
    y = np.concatenate(labels); parent = np.concatenate(parent_scores); final = np.concatenate(scores)
    return {"labels": y, "parent_scores": parent, "scores": final, "sample_ids": sample_ids,
            "node_count": np.concatenate(nodes), "outside_count": np.concatenate(outside),
            "parent_metrics": shared.binary_metrics(y, parent), "metrics": shared.binary_metrics(y, final)}


def evaluate_head(config: dict, head: str, data: RelaxedHeadData,
                  esm_cache: ESMResidueWindowCache, graph_cache: graph_train.GraphCache,
                  node_reader: SafeFullProteinNodeReader, device: torch.device,
                  data_identity: dict[str, str]) -> dict:
    output = ROOT / config["paths"]["output_root"] / head / "HieraPTM"
    metrics_path, complete = output / "metrics.json", output / "COMPLETE.flag"
    training = config["hieraptm_training"]
    seed = int(config["protocol"]["training_seed"])
    sequence_weight = ROOT / config["paths"]["output_root"] / head / "sequence_backbone/checkpoints/best.pt"
    identity = shared.canonical_sha256({"stage": "HieraPTM", "head": head, "seed": seed,
        "training": training, "data": data_identity,
        "sequence_backbone_weight_sha256": shared.sha256_file(sequence_weight),
        "config_sha256": config["_config_sha256"]})
    if complete.is_file() and metrics_path.is_file():
        result = json.loads(metrics_path.read_text("utf-8"))
        if result["identity_sha256"] != identity:
            raise ValueError(f"Completed HieraPTM identity mismatch: {head}")
        return result
    screen_path = output / "screen_metrics.json"
    if not (output / "SCREEN_COMPLETE.flag").is_file():
        raise FileNotFoundError(f"Training must finish before evaluation: {head}")
    screen = json.loads(screen_path.read_text("utf-8"))
    if screen["identity_sha256"] != identity:
        raise ValueError(f"Selected model identity mismatch: {head}")
    payload = torch.load(output / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    if payload["identity_sha256"] != identity:
        raise ValueError(f"Checkpoint identity mismatch: {head}")
    model = build_graph_variant("G4")
    model.cross_attention.dropout = float(training["cross_attention_dropout"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    fit, dev, test = indices(data, "fit"), indices(data, "development"), indices(data, "relaxed_test")
    collator = RelaxedGraphCollator(esm_cache, graph_cache, node_reader)
    dev_metrics = {"auroc": screen["development_auroc"], "auprc": screen["development_auprc"]}
    history = payload["history"]
    test_loader = make_loader(data, test, collator, training, seed, None)
    final = evaluate_final(model, test_loader, device)
    if final["sample_ids"] != [data.metadata[int(index)]["sample_id"] for index in test]:
        raise AssertionError("Relaxed test prediction order changed")
    extra = {"parent_sequence_backbone_score": final["parent_scores"], "graph_node_count": final["node_count"],
             "outside_sequence61_node_count": final["outside_count"]}
    rows = prediction_rows(data, test, final["labels"], final["scores"], "HieraPTM", extra)
    shared.atomic_tsv(rows, output / "relaxed_test_predictions.tsv")
    shared.write_curves(final["labels"], final["parent_scores"], output / "relaxed_test_curves/sequence_backbone")
    shared.write_curves(final["labels"], final["scores"], output / "relaxed_test_curves/HieraPTM")
    save_test_figure(output, head, final)
    graph_coverage = float((final["node_count"] > 0).mean())
    result = {"schema_version": 1, "status": "HieraPTM_RELAXED_TEST_COMPLETE", "identity_sha256": identity,
              "head": head, "ptm": config["heads"][head]["ptm"], "seed": seed,
              "fit_rows": len(fit), "development_rows": len(dev), "relaxed_test_rows": len(test),
              "development_auroc": dev_metrics["auroc"], "development_auprc": dev_metrics["auprc"],
              "sequence_backbone_relaxed_test_auroc": final["parent_metrics"]["auroc"],
              "sequence_backbone_relaxed_test_auprc": final["parent_metrics"]["auprc"],
              "relaxed_test_auroc": final["metrics"]["auroc"],
              "relaxed_test_auprc": final["metrics"]["auprc"],
              "delta_auroc_vs_sequence_backbone": final["metrics"]["auroc"] - final["parent_metrics"]["auroc"],
              "delta_auprc_vs_sequence_backbone": final["metrics"]["auprc"] - final["parent_metrics"]["auprc"],
              "paper_fig3_auroc": PAPER_AUROC[head],
              "delta_auroc_vs_paper_fig3": final["metrics"]["auroc"] - PAPER_AUROC[head],
              "graph_coverage": graph_coverage, "best_epoch": int(payload["epoch"]),
              "cross_gamma": float(model.cross_gamma.detach().cpu()),
              "gpu_hours": sum(float(row["epoch_seconds"]) for row in history) / 3600,
              "maximum_peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in history),
              "outer_test_used_for_selection": False, "outer_test_evaluations": 1,
              "formal_strict_test_rows_read": 0}
    shared.atomic_json(result, metrics_path); complete.write_text("COMPLETE\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    del model, test_loader; gc.collect(); torch.cuda.empty_cache()
    return result
