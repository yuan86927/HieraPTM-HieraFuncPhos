"""Final Task1 training dependencies, extracted without changing function bodies."""

from __future__ import annotations
import os
import time
from typing import Any
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
import utils as shared


ESM_MODELS = {"E2", "Sequence Backbone"}


def forward_model(
    model_name: str,
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    token_ids = batch["token_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    if model_name in ESM_MODELS:
        return model(
            token_ids,
            attention_mask,
            batch["esm_embeddings"].to(device, non_blocking=True),
            batch["esm_present_mask"].to(device, non_blocking=True),
        ).reshape(-1)
    return model(token_ids, attention_mask).reshape(-1)


def train_epoch(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pos_weight: torch.Tensor,
    training: dict[str, Any],
) -> dict[str, float | int]:
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    losses: list[float] = []
    classification_losses: list[float] = []
    auxiliary_losses: list[float] = []
    retries = 0
    samples = 0
    minimum_scale = float(scaler.get_scale())
    maximum_retries = int(training["maximum_amp_retries_per_batch"])
    process = psutil.Process(os.getpid())
    peak_host_rss = process.memory_info().rss / 1024**2
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for raw_batch in loader:
        target = raw_batch["label"].to(device, non_blocking=True).float().reshape(-1)
        samples += int(target.numel())
        resolved = False
        for _ in range(maximum_retries + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = forward_model(model_name, model, raw_batch, device)
                classification_loss = criterion(logits, target)
                auxiliary_loss = (
                    model.moe_auxiliary_loss()
                    if hasattr(model, "moe_auxiliary_loss")
                    else classification_loss.new_zeros(())
                )
                loss = classification_loss + auxiliary_loss
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite loss/logits cannot be resolved")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if bool(torch.isfinite(gradient_norm)):
                scaler.step(optimizer)
                scaler.update()
                resolved = True
                break
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            retries += 1
            minimum_scale = min(minimum_scale, scale_after)
            if scale_after >= scale_before:
                raise FloatingPointError("GradScaler failed to back off")
        if not resolved:
            raise FloatingPointError("Nonfinite gradients persisted after AMP retries")
        losses.append(float(loss.detach().cpu()))
        classification_losses.append(float(classification_loss.detach().cpu()))
        auxiliary_losses.append(float(auxiliary_loss.detach().cpu()))
        peak_host_rss = max(peak_host_rss, process.memory_info().rss / 1024**2)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "train_loss": float(np.mean(losses)),
        "train_classification_loss": float(np.mean(classification_losses)),
        "train_moe_auxiliary_loss": float(np.mean(auxiliary_losses)),
        "steps": len(losses),
        "samples": samples,
        "seconds": seconds,
        "samples_per_second": samples / seconds,
        "amp_retries": retries,
        "minimum_amp_scale": minimum_scale,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "peak_host_rss_mib": peak_host_rss,
    }


def evaluate(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    collect_gate: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    gate_positions: list[np.ndarray] = []
    gate_centers: list[np.ndarray] = []
    gate_locals: list[np.ndarray] = []
    gate_masks: list[np.ndarray] = []
    weighted_loss_sum = 0.0
    rows = 0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")
    process = psutil.Process(os.getpid())
    peak_host_rss = process.memory_info().rss / 1024**2
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            target = raw_batch["label"].to(device, non_blocking=True).float().reshape(-1)
            if collect_gate:
                token_ids = raw_batch["token_ids"].to(device, non_blocking=True)
                attention_mask = raw_batch["attention_mask"].to(device, non_blocking=True)
                esm_embeddings = raw_batch["esm_embeddings"].to(device, non_blocking=True)
                esm_present_mask = raw_batch["esm_present_mask"].to(device, non_blocking=True)
                representation = model.encode(
                    token_ids, attention_mask, esm_embeddings, esm_present_mask
                )
                logits = model.backbone.classifier(representation["center_token"])[:, 0]
                gate_positions.append(
                    representation["esm_gate_position_mean"].float().cpu().numpy()
                )
                gate_centers.append(
                    representation["esm_gate_center_mean"].float().cpu().numpy()
                )
                gate_locals.append(
                    representation["esm_gate_local_mean"].float().cpu().numpy()
                )
                gate_masks.append(esm_present_mask.bool().cpu().numpy())
            else:
                logits = forward_model(model_name, model, raw_batch, device)
            weighted_loss_sum += float(criterion(logits, target).item())
            rows += int(target.numel())
            labels.append(target.cpu().numpy())
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            peak_host_rss = max(peak_host_rss, process.memory_info().rss / 1024**2)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    all_labels = np.concatenate(labels)
    all_scores = np.concatenate(scores)
    metrics = shared.binary_metrics(all_labels, all_scores)
    metrics.update({
        "loss": weighted_loss_sum / rows,
        "seconds": seconds,
        "samples_per_second": rows / seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "peak_host_rss_mib": peak_host_rss,
    })
    gate_payload = None
    if gate_positions:
        gate_payload = {
            "position": np.concatenate(gate_positions),
            "center": np.concatenate(gate_centers),
            "local": np.concatenate(gate_locals),
            "present_mask": np.concatenate(gate_masks),
        }
    return metrics, all_labels, all_scores, gate_payload
