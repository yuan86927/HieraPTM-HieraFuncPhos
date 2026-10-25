"""Training state, optimizer/scaler state and RNG checkpoint persistence."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
import utils as shared


def save_checkpoint(path: Path, identity: str, model: torch.nn.Module, optimizer: Any,
                    scaler: Any, epoch: int, best: dict[str, Any], patience: int,
                    history: list[dict[str, Any]], scheduler: Any | None = None) -> None:
    payload = {
        "schema_version": 1, "identity_sha256": identity, "epoch": epoch,
        "best_metrics": best, "patience_counter": patience, "history": history,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(), "rng_state": shared.capture_rng_state(),
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    shared.save_checkpoint(payload, path)


def load_checkpoint(path: Path, identity: str, model: torch.nn.Module, optimizer: Any,
                    scaler: Any, device: torch.device, scheduler: Any | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("identity_sha256") != identity:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    model.load_state_dict(payload["model_state"]); optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload["scaler_state"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    shared.restore_rng_state(payload["rng_state"])
    return payload
