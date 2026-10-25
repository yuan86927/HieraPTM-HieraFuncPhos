"""Sequence Backbone and HieraPTM training; validation selection only."""
from __future__ import annotations
import gc
import json
import math
from typing import Any
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import utils as shared
import sequence_optimization as sequence_training
import structure_optimization as ca2_train
import graph_data as graph_train
from configuration import ROOT
from models import SequenceBackbone, build_graph_variant
from esm_cache import ESMResidueWindowCache
from dataset import FinalSplitData as RelaxedHeadData
from data import MissingSafeESMCollator, SafeFullProteinNodeReader, RelaxedGraphCollator, indices, make_loader
from checkpoints import save_checkpoint, load_checkpoint
from reporting import prediction_rows, plot_training


def train_sequence_backbone(config: dict[str, Any], head: str, data: RelaxedHeadData,
             esm_cache: ESMResidueWindowCache, device: torch.device,
             data_identity: dict[str, str]) -> dict[str, Any]:
    output = ROOT / config["paths"]["output_root"] / head / "sequence_backbone"
    metrics_path, complete = output / "metrics.json", output / "COMPLETE.flag"
    training = config["sequence_backbone_training"]; seed = int(config["protocol"]["training_seed"])
    identity_payload = {"stage": "Sequence Backbone", "head": head, "seed": seed, "training": training,
                        "data": data_identity, "config_sha256": config["_config_sha256"]}
    identity = shared.canonical_sha256(identity_payload)
    if complete.is_file() and metrics_path.is_file():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        if result["identity_sha256"] != identity: raise ValueError(f"Completed Sequence Backbone identity mismatch: {head}")
        print(json.dumps({"status": "SKIPPED_COMPLETE_SEQUENCE_BACKBONE", "head": head, "dev_auroc": result["development_auroc"]}))
        return result
    output.mkdir(parents=True, exist_ok=True)
    fit, dev = indices(data, "fit"), indices(data, "development")
    fit_labels = data.labels[fit]; positives = int(fit_labels.sum()); negatives = len(fit) - positives
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    collator = MissingSafeESMCollator(esm_cache)
    dev_loader = make_loader(data, dev, collator, training, seed, None)
    shared.seed_everything(seed); model = SequenceBackbone().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=float(training["amp_initial_scale"]),
                                  backoff_factor=float(training["amp_backoff_factor"]))
    last, best_path = output / "checkpoints/last.pt", output / "checkpoints/best.pt"
    start = patience = 0; best = None; history: list[dict[str, Any]] = []
    if last.is_file():
        payload = load_checkpoint(last, identity, model, optimizer, scaler, device)
        start = int(payload["epoch"]) + 1; patience = int(payload["patience_counter"])
        best = dict(payload["best_metrics"]); history = list(payload["history"])
    for epoch in range(start, int(training["max_epochs"])):
        fit_loader = make_loader(data, fit, collator, training, seed, epoch)
        train = sequence_training.train_epoch("Sequence Backbone", model, fit_loader, optimizer, scaler, device, pos_weight, training)
        valid, _, _, _ = sequence_training.evaluate("Sequence Backbone", model, dev_loader, device, pos_weight)
        improved = shared.is_better(valid, best); patience = 0 if improved else patience + 1
        if improved: best = dict(valid)
        row = {"epoch": epoch, "train_loss": train["train_loss"], "validation_loss": valid["loss"],
               "auroc": valid["auroc"], "auprc": valid["auprc"],
               "train_seconds": train["seconds"], "validation_seconds": valid["seconds"],
               "epoch_seconds": train["seconds"] + valid["seconds"],
               "training_samples_per_second": train["samples_per_second"],
               "peak_allocated_mib": max(train["peak_allocated_mib"], valid["peak_allocated_mib"]),
               "patience_counter": patience, "improved": improved,
               "amp_retries": train["amp_retries"]}
        history = [item for item in history if int(item["epoch"]) != epoch] + [row]
        history.sort(key=lambda value: int(value["epoch"])); shared.atomic_tsv(history, output / "training_log.tsv")
        save_checkpoint(last, identity, model, optimizer, scaler, epoch, best, patience, history)
        if improved: save_checkpoint(best_path, identity, model, optimizer, scaler, epoch, best, patience, history)
        print(json.dumps({"stage": "Sequence Backbone", "head": head, "epoch": epoch, "auroc": valid["auroc"],
                          "auprc": valid["auprc"], "patience": patience}), flush=True)
        del fit_loader
        if patience >= int(training["early_stopping_patience"]): break
    payload = load_checkpoint(best_path, identity, model, optimizer, scaler, device)
    valid, labels, scores, _ = sequence_training.evaluate("Sequence Backbone", model, dev_loader, device, pos_weight)
    shared.atomic_tsv(prediction_rows(data, dev, labels, scores, "Sequence Backbone"), output / "development_predictions.tsv")
    shared.write_curves(labels, scores, output / "development_curves")
    plot_training(history, output / "training_curves.png", f"{head} Sequence Backbone relaxed training")
    result = {"schema_version": 1, "status": "SEQUENCE_BACKBONE_DEVELOPMENT_COMPLETE", "identity_sha256": identity,
              "head": head, "seed": seed, "fit_rows": len(fit), "development_rows": len(dev),
              "development_auroc": valid["auroc"], "development_auprc": valid["auprc"],
              "best_epoch": int(payload["epoch"]),
              "gpu_hours": sum(float(row["epoch_seconds"]) for row in history) / 3600,
              "maximum_peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in history),
              "relaxed_test_rows_read": 0, "formal_strict_test_rows_read": 0}
    shared.atomic_json(result, metrics_path); complete.write_text("COMPLETE\n", encoding="utf-8")
    if last.is_file(): last.unlink()
    del model, optimizer, scaler, dev_loader; gc.collect(); torch.cuda.empty_cache()
    return result


def train_hieraptm(config: dict[str, Any], head: str, data: RelaxedHeadData,
               esm_cache: ESMResidueWindowCache, graph_cache: graph_train.GraphCache,
               node_reader: SafeFullProteinNodeReader, device: torch.device,
               data_identity: dict[str, str]) -> dict[str, Any]:
    output = ROOT / config["paths"]["output_root"] / head / "HieraPTM"
    metrics_path, complete = output / "metrics.json", output / "COMPLETE.flag"
    screen_metrics_path = output / "screen_metrics.json"
    screen_complete = output / "SCREEN_COMPLETE.flag"
    sequence_backbone_root = ROOT / config["paths"]["output_root"] / head / "sequence_backbone"
    if not (sequence_backbone_root / "COMPLETE.flag").is_file(): raise RuntimeError(f"Sequence Backbone must complete first: {head}")
    sequence_backbone_weight = sequence_backbone_root / "checkpoints/best.pt"
    training = config["hieraptm_training"]; seed = int(config["protocol"]["training_seed"])
    identity_payload = {"stage": "HieraPTM", "head": head, "seed": seed, "training": training,
                        "data": data_identity, "sequence_backbone_weight_sha256": shared.sha256_file(sequence_backbone_weight),
                         "config_sha256": config["_config_sha256"]}
    identity = shared.canonical_sha256(identity_payload); output.mkdir(parents=True, exist_ok=True)
    if complete.is_file() and metrics_path.is_file():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        if result.get("identity_sha256") != identity:
            raise ValueError(f"Completed HieraPTM identity mismatch: {head}")
        print(json.dumps({"status": "SKIPPED_COMPLETE_HieraPTM", "head": head,
                          "test_auroc": result["relaxed_test_auroc"]}))
        return result
    if screen_complete.is_file() and screen_metrics_path.is_file():
        result = json.loads(screen_metrics_path.read_text(encoding="utf-8"))
        if result.get("identity_sha256") != identity:
            raise ValueError(f"Completed HieraPTM screen identity mismatch: {head}")
        print(json.dumps({"status": "SKIPPED_COMPLETE_HieraPTM_SCREEN", "head": head,
                          "development_auroc": result["development_auroc"]}))
        return result
    fit, dev = indices(data, "fit"), indices(data, "development")
    fit_labels = data.labels[fit]; positives = int(fit_labels.sum()); negatives = len(fit) - positives
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    collator = RelaxedGraphCollator(esm_cache, graph_cache, node_reader)
    dev_loader = make_loader(data, dev, collator, training, seed, None)
    shared.seed_everything(seed); model = build_graph_variant("G4").to(device)
    model.cross_attention.dropout = float(training["cross_attention_dropout"])
    parent_payload = torch.load(sequence_backbone_weight, map_location="cpu", weights_only=False)
    model.load_parent_state(parent_payload["model_state"])
    optimizer, optimizer_rows = ca2_train.build_optimizer(
        model, training["learning_rates"], float(training["weight_decay"])
    )
    steps_per_epoch = math.ceil(len(fit) / int(training["batch_size"]))
    total_steps = steps_per_epoch * int(training["max_epochs"])
    warmup_steps = max(1, round(total_steps * float(training["warmup_fraction"])))
    scheduler = LambdaLR(optimizer, lambda step: ca2_train.schedule_multiplier(
        step, total_steps, warmup_steps, float(training["minimum_lr_ratio"])
    ))
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=float(training["amp_initial_scale"]),
                                  backoff_factor=float(training["amp_backoff_factor"]))
    shared.atomic_tsv(optimizer_rows, output / "optimizer_parameter_groups.tsv")
    first = graph_train.move(next(iter(dev_loader)), device); model.eval()
    with torch.inference_mode():
        parent = model.parent(first["token_ids"], first["attention_mask"], first["esm_embeddings"], first["esm_present_mask"])
        initial = graph_train.model_logits(model, first)
    initial_difference = float((parent.reshape(-1) - initial).abs().max().cpu())
    if initial_difference != 0.0: raise AssertionError(f"HieraPTM is not identity-initialized: {initial_difference}")
    del first, parent, initial; torch.cuda.empty_cache()
    last, best_path = output / "checkpoints/last.pt", output / "checkpoints/best.pt"
    start = patience = 0; best = None; history: list[dict[str, Any]] = []
    if last.is_file():
        payload = load_checkpoint(last, identity, model, optimizer, scaler, device, scheduler)
        start = int(payload["epoch"]) + 1; patience = int(payload["patience_counter"])
        best = dict(payload["best_metrics"]); history = list(payload["history"])
    for epoch in range(start, int(training["max_epochs"])):
        fit_loader = make_loader(data, fit, collator, training, seed, epoch)
        train = ca2_train.train_epoch(model, fit_loader, optimizer, scheduler, scaler, device, pos_weight, training)
        valid, _, _, _ = graph_train.evaluate(model, dev_loader, device, pos_weight)
        improved = shared.is_better(valid, best); patience = 0 if improved else patience + 1
        if improved: best = dict(valid)
        row = {"epoch": epoch, "train_loss": train["train_loss"], "train_bce": train["train_bce"],
               "train_residual_mse": train["train_residual_mse"], "validation_loss": valid["loss"],
               "auroc": valid["auroc"], "auprc": valid["auprc"],
               "train_seconds": train["seconds"], "validation_seconds": valid["seconds"],
               "epoch_seconds": train["seconds"] + valid["seconds"],
               "training_samples_per_second": train["samples_per_second"],
               "peak_allocated_mib": max(train["peak_allocated_mib"], valid["peak_allocated_mib"]),
               "patience_counter": patience, "improved": improved,
               "cross_gamma": float(model.cross_gamma.detach().cpu()),
               **{key: value for key, value in train.items() if key.startswith("lr_")}}
        history = [item for item in history if int(item["epoch"]) != epoch] + [row]
        history.sort(key=lambda value: int(value["epoch"])); shared.atomic_tsv(history, output / "training_log.tsv")
        save_checkpoint(last, identity, model, optimizer, scaler, epoch, best, patience, history, scheduler)
        if improved: save_checkpoint(best_path, identity, model, optimizer, scaler, epoch, best, patience, history, scheduler)
        print(json.dumps({"stage": "HieraPTM", "head": head, "epoch": epoch, "auroc": valid["auroc"],
                          "auprc": valid["auprc"], "cross_gamma": row["cross_gamma"],
                          "patience": patience}), flush=True)
        del fit_loader
        if patience >= int(training["early_stopping_patience"]): break
    payload = load_checkpoint(best_path, identity, model, optimizer, scaler, device, scheduler)
    dev_metrics, dev_y, dev_score, _ = graph_train.evaluate(model, dev_loader, device, pos_weight)
    shared.write_curves(dev_y, dev_score, output / "development_curves")
    plot_training(history, output / "training_curves.png", f"{head} HieraPTM relaxed training")

    result = {
        "schema_version": 2,
        "status": "HieraPTM_DEVELOPMENT_SCREEN_COMPLETE",
        "identity_sha256": identity,
        "head": head,
        "ptm": config["heads"][head]["ptm"],
        "seed": seed,
        "fit_rows": len(fit),
        "development_rows": len(dev),
        "development_auroc": dev_metrics["auroc"],
        "development_auprc": dev_metrics["auprc"],
        "best_epoch": int(payload["epoch"]),
        "gpu_hours": sum(float(row["epoch_seconds"]) for row in history) / 3600,
        "maximum_peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in history),
        "outer_test_evaluated": False,
        "outer_test_used_for_selection": False,
        "formal_strict_test_rows_read": 0,
    }
    shared.atomic_json(result, screen_metrics_path)
    screen_complete.write_text("COMPLETE\n", encoding="utf-8")
    if last.is_file():
        last.unlink()
    print(json.dumps(result, ensure_ascii=False), flush=True)
    del model, optimizer, scaler, scheduler, dev_loader
    gc.collect(); torch.cuda.empty_cache()
    return result
