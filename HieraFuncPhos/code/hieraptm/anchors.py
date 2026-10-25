"""Restore the selected functional reader and extract only requested roles."""
from __future__ import annotations

import numpy as np
import torch

from .data import make_loader
from .dynamic_math import tensor_sha
from .models.anchors import InstrumentedBase, base_forward
from .utils import move, sha


def extract(cache, args, weights, checkpoint_path, contract, roles, device):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved["contract"] != contract:
        raise RuntimeError("Selected functional checkpoint does not match this run")
    model = InstrumentedBase(weights["ST"], weights["Y"], **saved["model_config"])
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    if set(saved["adapter_state"]) != expected or set(saved["trainable_names"]) != expected:
        raise RuntimeError("Functional checkpoint adapter parameter schema differs")
    loaded = model.load_state_dict(saved["adapter_state"], strict=False)
    if loaded.unexpected_keys:
        raise RuntimeError(f"Unexpected functional checkpoint parameters: {loaded.unexpected_keys}")
    model.requires_grad_(False).to(device).eval()
    count = len(cache.index)
    # Unrequested anchors are NaN, not plausible zeros. Preserve the source's
    # canonical full-index batch context, then retain ONLY requested roles.
    # Other-role inputs may share a forward batch, but their labels never affect
    # forwarding/batching, and their outputs are not retained or scored here.
    arrays = {"functional": np.full((count, 192), np.nan, np.float32),
              "static_delta": np.full((count, 192), np.nan, np.float32),
              "reference_logit": np.full(count, np.nan, np.float32)}
    retained = np.zeros(count, dtype=bool)
    retained[np.concatenate([cache.roles[role] for role in roles])] = True
    ids = np.arange(count, dtype=np.int64)
    with torch.inference_mode():
        for raw in make_loader(cache, ids, args, False, args.seed, workers=0):
            row_ids = raw["row_id"].numpy()
            keep = retained[row_ids]
            if not keep.any():
                continue
            result = base_forward(model, move(raw, device))
            for key, output in (("functional", "functional"), ("static_delta", "static_delta"),
                                ("reference_logit", "logits")):
                value = result[output].float().cpu().numpy()
                if not np.isfinite(value).all():
                    raise RuntimeError(f"Nonfinite frozen anchor: {key}")
                arrays[key][row_ids[keep]] = value[keep]
    head = {key: value.detach().cpu().clone() for key, value in model.functional_head.state_dict().items()}
    if len(head) != 6:
        raise RuntimeError("Expected the six-tensor frozen functional head")
    anchors = {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}
    return anchors, head, {"functional_checkpoint": sha(checkpoint_path), "frozen_head": tensor_sha(head),
                           "extracted_roles": list(roles), "precision": "float32",
                           "batch_context": "full cache index order; retain requested roles only",
                           "batch_size": args.batch_size, "labels_used_in_forward_or_batching": False}
