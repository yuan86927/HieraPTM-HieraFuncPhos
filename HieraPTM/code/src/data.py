"""Final split validation, feature batches and deterministic data loaders."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils as shared
import graph_data as graph_train
from configuration import resolve
from dataset import FinalSplitData
from dataset import FinalSplitData as RelaxedHeadData
from esm_cache import ESMResidueWindowCache


SPLIT_NAME = {0: "fit", 1: "development", 2: "relaxed_test"}


class MissingSafeESMCollator:
    """Return zero/missing ESM values for the tiny set absent from the frozen cache."""

    def __init__(self, cache: ESMResidueWindowCache) -> None:
        self.cache = cache

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        sample_ids = [row["sample_id"] for row in rows]
        embeddings = np.zeros((len(rows), 61, 640), dtype=np.float16)
        present = np.zeros((len(rows), 61), dtype=np.bool_)
        available_positions = [index for index, sample_id in enumerate(sample_ids) if sample_id in self.cache.index]
        if available_positions:
            identifiers = [sample_ids[index] for index in available_positions]
            values, masks = self.cache.get_batch(identifiers, output_dtype=np.float16)
            embeddings[available_positions] = values
            present[available_positions] = masks
        attention = np.stack([row["attention_mask"] for row in rows]).astype(np.bool_)
        present &= attention
        return {
            "sample_id": sample_ids,
            "protein_id": [row["protein_id"] for row in rows],
            "ptm": [row["ptm"] for row in rows],
            "head": [row["head"] for row in rows],
            "sequence_61": [row["sequence_61"] for row in rows],
            "split": [row["split"] for row in rows],
            "token_ids": torch.from_numpy(np.stack([row["token_ids"] for row in rows]).astype(np.int64)),
            "attention_mask": torch.from_numpy(attention),
            "esm_embeddings": torch.from_numpy(embeddings),
            "esm_present_mask": torch.from_numpy(present),
            "label": torch.from_numpy(np.asarray([row["label"] for row in rows], dtype=np.float32)),
            "fold_id": torch.from_numpy(np.asarray([row["fold_id"] for row in rows], dtype=np.int64)),
        }


class SafeFullProteinNodeReader(graph_train.FullProteinNodeReader):
    def get_batch(self, protein_ids: list[str], positions: np.ndarray, mask: np.ndarray) -> np.ndarray:
        output = np.zeros((*positions.shape, 640), dtype=np.float16)
        by_shard: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for batch_index, protein_id in enumerate(protein_ids):
            valid_slots = np.flatnonzero(mask[batch_index])
            if not len(valid_slots):
                continue
            if protein_id not in self.proteins:
                raise KeyError(f"Graph-present protein absent from ESM index: {protein_id}")
            meta = self.proteins[protein_id]
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


class RelaxedGraphCollator:
    def __init__(self, esm_cache: ESMResidueWindowCache, graph_cache: graph_train.GraphCache,
                 node_reader: SafeFullProteinNodeReader) -> None:
        self.sequence = MissingSafeESMCollator(esm_cache)
        self.graph_cache = graph_cache
        self.node_reader = node_reader

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.sequence(rows)
        positions, coordinates, geometry, mask = self.graph_cache.get_batch(batch["sample_id"])
        node_esm = self.node_reader.get_batch(batch["protein_id"], positions, mask)
        batch.update({
            "node_esm": torch.from_numpy(node_esm),
            "node_positions": torch.from_numpy(positions.astype(np.int64)),
            "node_coordinates": torch.from_numpy(coordinates),
            "node_geometry": torch.from_numpy(geometry),
            "node_mask": torch.from_numpy(mask),
        })
        return batch


def indices(data: RelaxedHeadData, split: str) -> np.ndarray:
    code = {value: key for key, value in SPLIT_NAME.items()}[split]
    return np.flatnonzero(data.fold_ids == code)


def make_loader(data: RelaxedHeadData, selected: np.ndarray, collator: Any,
                training: dict[str, Any], seed: int, epoch: int | None) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": shared.FusionIndexedDataset(data, selected),
        "batch_size": int(training["batch_size"]), "num_workers": 0,
        "pin_memory": bool(training["pin_memory"]), "collate_fn": collator,
    }
    if epoch is None:
        kwargs["shuffle"] = False
    else:
        kwargs["sampler"] = shared.EpochSampler(len(selected), seed, epoch)
    return DataLoader(**kwargs)


def checked_data(config: dict, head: str) -> tuple[FinalSplitData, dict]:
    path = resolve(config["paths"]["split_directory"]) / f"{head}.tsv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Set paths.split_directory to the final split tables: {path}")
    digest = shared.sha256_file(path)
    manifest = path.parent / "manifest.json"
    if manifest.is_file():
        published = json.loads(manifest.read_text("utf-8"))
        entry, = [item for item in published["heads"] if item["head"] == head]
        if entry["sha256"] != digest:
            raise ValueError(f"Published split table hash differs: {head}")
    data = FinalSplitData(path, head, config["heads"][head])
    return data, {"final_split_table_sha256": digest,
                  "final_split_code_sha256": hashlib.sha256(data.fold_ids.tobytes()).hexdigest()}


def feature_contract(config: dict) -> dict:
    paths = config["paths"]
    files = [resolve(paths[key]) for key in
             ("esm_cache_registry", "prepared_site_index", "prepared_protein_index")]
    graph_root = resolve(paths["graph_cache_root"])
    files += [graph_root / name for name in (
        "manifest.json", "READY.flag", "sample_index.tsv", "node_positions_1based.int32.npy",
        "node_centered_ca.float16.npy", "node_geometry.float16.npy", "node_mask.bool.npy")]
    missing = [str(path) for path in files if not path.is_file()]
    if not resolve(paths["esm_cache_root"]).is_dir():
        missing.append(str(resolve(paths["esm_cache_root"])))
    if missing:
        raise FileNotFoundError("Configure the prepared feature inputs:\n" + "\n".join(missing))
    return {"esm_registry_sha256": shared.sha256_file(resolve(paths["esm_cache_registry"])),
            "site_index_sha256": shared.sha256_file(resolve(paths["prepared_site_index"])),
            "protein_index_sha256": shared.sha256_file(resolve(paths["prepared_protein_index"])),
            "graph_manifest_sha256": shared.sha256_file(graph_root / "manifest.json")}
