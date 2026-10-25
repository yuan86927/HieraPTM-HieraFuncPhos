"""Final Task1 training dependencies, extracted without changing function bodies."""

from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils as shared
from esm_cache import ESMResidueWindowCache, close_memmap


class GraphCache:
    def __init__(self, root: Path) -> None:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "READY_FOR_G1" or not (root / "READY.flag").is_file():
            raise RuntimeError("G1 graph cache is not sealed")
        self.positions = np.load(root / "node_positions_1based.int32.npy", mmap_mode="r")
        self.coordinates = np.load(root / "node_centered_ca.float16.npy", mmap_mode="r")
        self.geometry = np.load(root / "node_geometry.float16.npy", mmap_mode="r")
        self.masks = np.load(root / "node_mask.bool.npy", mmap_mode="r")
        with (root / "sample_index.tsv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.index = {row["sample_id"]: int(row["row_id"]) for row in rows}
        if len(self.index) != len(rows) or self.positions.shape != (len(rows), 33):
            raise AssertionError("Graph cache/index contract mismatch")

    def get_batch(self, sample_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.asarray([self.index[sample_id] for sample_id in sample_ids], dtype=np.int64)
        return (np.asarray(self.positions[indices], dtype=np.int32),
                np.asarray(self.coordinates[indices], dtype=np.float16),
                np.asarray(self.geometry[indices], dtype=np.float16),
                np.asarray(self.masks[indices], dtype=np.bool_))

    def close(self) -> None:
        for array in (self.positions, self.coordinates, self.geometry, self.masks): close_memmap(array)


class FullProteinNodeReader:
    def __init__(self, protein_index: Path, esm_cache: ESMResidueWindowCache) -> None:
        with protein_index.open("r", encoding="utf-8", newline="") as handle:
            self.proteins = {row["protein_id"]: {"length": int(row["protein_length"]),
                                                 "shard": int(row["cache_shard_id"]),
                                                 "offset": int(row["protein_shard_offset0"])}
                             for row in csv.DictReader(handle, delimiter="\t")}
        self.esm_cache = esm_cache

    def get_batch(self, protein_ids: list[str], positions: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
        output = np.zeros((*positions.shape, 640), dtype=np.float16)
        by_shard: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for batch_index, protein_id in enumerate(protein_ids):
            meta = self.proteins[protein_id]
            valid_slots = np.flatnonzero(mask[batch_index])
            valid_positions = positions[batch_index, valid_slots]
            if np.any(valid_positions < 1) or np.any(valid_positions > meta["length"]):
                raise IndexError(f"Graph node outside protein: {protein_id}")
            absolute = meta["offset"] + valid_positions - 1
            by_shard.setdefault(meta["shard"], []).append((batch_index, valid_slots, absolute))
        for shard in sorted(by_shard):
            array = self.esm_cache._mmap(shard)
            for batch_index, slots, absolute in by_shard[shard]:
                output[batch_index, slots] = np.asarray(array[absolute], dtype=np.float16)
        return output


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def model_logits(model: Any, batch: dict[str, Any]) -> torch.Tensor:
    return model(batch["token_ids"], batch["attention_mask"], batch["esm_embeddings"],
                 batch["esm_present_mask"], batch["node_esm"], batch["node_positions"],
                 batch["node_coordinates"], batch["node_geometry"], batch["node_mask"]).reshape(-1)


def evaluate(model: Any, stream: DataLoader, device: torch.device,
             pos_weight: torch.Tensor, collect: bool = False) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    model.eval(); labels = []; scores = []; sample_ids = []; node_counts = []; outside_counts = []
    pool_weights = []; gate_means = []; layer_attentions: list[list[np.ndarray]] = []; cross_attentions = []
    node_geometry = []; weighted_loss = 0.0; rows = 0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); started = time.perf_counter()
    with torch.inference_mode():
        for raw in stream:
            batch = move(raw, device)
            representation = model.encode(batch["token_ids"], batch["attention_mask"],
                batch["esm_embeddings"], batch["esm_present_mask"], batch["node_esm"],
                batch["node_positions"], batch["node_coordinates"], batch["node_geometry"], batch["node_mask"])
            prediction = model.parent.backbone.classifier(representation["center_token"])[:, 0]
            target = batch["label"].float().reshape(-1)
            weighted_loss += float(criterion(prediction, target).item()); rows += len(target)
            labels.append(target.cpu().numpy()); scores.append(torch.sigmoid(prediction).float().cpu().numpy())
            sample_ids.extend(raw["sample_id"]); node_counts.append(raw["node_mask"].sum(1).numpy())
            outside_counts.append((raw["node_geometry"][:, :, 7] * raw["node_mask"]).sum(1).numpy())
            if collect:
                pool_weights.append(representation["graph_pool_weights"].float().cpu().numpy())
                gate_means.append(representation["fusion_gate"].float().mean(1).cpu().numpy())
                node_geometry.append(raw["node_geometry"].numpy())
                if not layer_attentions: layer_attentions = [[] for _ in representation["layer_center_attentions"]]
                for index, values in enumerate(representation["layer_center_attentions"]):
                    layer_attentions[index].append(values.float().cpu().numpy())
                cross = representation.get("cross_attention_weights")
                if cross is not None:
                    cross_attentions.append(cross[:, :, 30].float().cpu().numpy())
    torch.cuda.synchronize(device); seconds = time.perf_counter() - started
    y = np.concatenate(labels); score = np.concatenate(scores); metrics = shared.binary_metrics(y, score)
    metrics.update({"loss": weighted_loss / rows, "seconds": seconds, "samples_per_second": rows / seconds,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2})
    diagnostics = {"sample_ids": sample_ids, "node_count": np.concatenate(node_counts),
                   "outside_count": np.concatenate(outside_counts),
                   "pool_weights": np.concatenate(pool_weights) if pool_weights else np.zeros((rows, 33)),
                   "gate_mean": np.concatenate(gate_means) if gate_means else np.zeros(rows),
                   "node_geometry": np.concatenate(node_geometry) if node_geometry else np.zeros((rows, 33, 8)),
                   "layer_attentions": [np.concatenate(values) for values in layer_attentions],
                   "cross_center_attentions": np.concatenate(cross_attentions) if cross_attentions else None}
    return metrics, y, score, diagnostics
