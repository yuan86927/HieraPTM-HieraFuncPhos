"""Final Task1 training dependencies, extracted without changing function bodies."""

from __future__ import annotations
import math
import os
import time
from typing import Any
import numpy as np
import psutil
import torch
import graph_data as graph_train


def logits_and_residual(model: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    representation = model.encode(
        batch["token_ids"], batch["attention_mask"], batch["esm_embeddings"],
        batch["esm_present_mask"], batch["node_esm"], batch["node_positions"],
        batch["node_coordinates"], batch["node_geometry"], batch["node_mask"],
    )
    classifier = model.parent.backbone.classifier
    logits = classifier(representation["center_token"])[:, 0]
    with torch.no_grad():
        parent_logits = classifier(representation["sequence_center_token"])[:, 0]
    return logits, logits - parent_logits


def parameter_family(name: str) -> str:
    if name == "cross_gamma" or name.startswith("cross_attention.out_proj") or name.startswith("cross_output_norm"):
        return "gate_projection"
    if name.startswith("cross_") or name.startswith("cross_attention.in_proj"):
        return "cross_attention"
    if name.startswith(("geometry_adapter", "node_input_norm", "graph_layers", "graph_final_norm")):
        return "graph_encoder"
    raise AssertionError(f"Unexpected trainable G4 parameter: {name}")


def build_optimizer(model: Any, profile: dict[str, float], weight_decay: float) -> tuple[Any, list[dict[str, Any]]]:
    grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    audit: dict[tuple[str, bool], list[str]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = parameter_family(name)
        no_decay = name.endswith(".bias") or "norm" in name.lower() or name.endswith("gamma")
        grouped.setdefault((family, no_decay), []).append(parameter)
        audit.setdefault((family, no_decay), []).append(name)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assigned = {id(p) for values in grouped.values() for p in values}
    if assigned != expected:
        raise AssertionError("CA2 optimizer did not assign every trainable parameter exactly once")
    groups = []
    audit_rows = []
    for (family, no_decay), parameters in sorted(grouped.items()):
        lr = float(profile[family]); decay = 0.0 if no_decay else float(weight_decay)
        groups.append({"params": parameters, "lr": lr, "initial_lr": lr,
                       "weight_decay": decay, "group_name": f"{family}_{'no_decay' if no_decay else 'decay'}"})
        audit_rows.append({"family": family, "decay_policy": "none" if no_decay else "adamw",
                           "learning_rate": lr, "weight_decay": decay,
                           "parameter_tensors": len(parameters),
                           "parameters": sum(p.numel() for p in parameters),
                           "names": ";".join(audit[(family, no_decay)])})
    return torch.optim.AdamW(groups), audit_rows


def schedule_multiplier(step: int, total_steps: int, warmup_steps: int, minimum_ratio: float) -> float:
    if step < warmup_steps:
        return max(1, step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress))


def train_epoch(model: Any, stream: Any, optimizer: Any, scheduler: Any, scaler: Any,
                device: torch.device, pos_weight: torch.Tensor, training: dict[str, Any]) -> dict[str, Any]:
    model.train(); criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    losses, bce_losses, residual_losses = [], [], []
    retries = samples = 0; process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss / 1024**2
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); started = time.perf_counter()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    residual_lambda = float(training["residual_logit_lambda"])
    for raw in stream:
        batch = graph_train.move(raw, device); target = batch["label"].float().reshape(-1); samples += len(target)
        resolved = False
        for _ in range(int(training["maximum_amp_retries_per_batch"]) + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction, residual = logits_and_residual(model, batch)
                bce = criterion(prediction, target)
                residual_penalty = residual.float().square().mean()
                loss = bce + residual_lambda * residual_penalty
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite CA2 loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
            if bool(torch.isfinite(norm)):
                scaler.step(optimizer); scaler.update(); scheduler.step(); resolved = True; break
            before = float(scaler.get_scale()); scaler.step(optimizer); scaler.update(); retries += 1
            if float(scaler.get_scale()) >= before:
                raise FloatingPointError("CA2 AMP scaler did not back off")
        if not resolved:
            raise FloatingPointError("Nonfinite CA2 gradients persisted")
        losses.append(float(loss.detach().cpu())); bce_losses.append(float(bce.detach().cpu()))
        residual_losses.append(float(residual_penalty.detach().cpu()))
        peak_rss = max(peak_rss, process.memory_info().rss / 1024**2)
    torch.cuda.synchronize(device); seconds = time.perf_counter() - started
    family_lrs = {}
    for group in optimizer.param_groups:
        family_lrs[group["group_name"].split("_no_decay")[0].split("_decay")[0]] = float(group["lr"])
    return {"train_loss": float(np.mean(losses)), "train_bce": float(np.mean(bce_losses)),
            "train_residual_mse": float(np.mean(residual_losses)), "steps": len(losses),
            "seconds": seconds, "samples_per_second": samples / seconds, "amp_retries": retries,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            "peak_host_rss_mib": peak_rss, **{f"lr_{key}": value for key, value in family_lrs.items()}}
