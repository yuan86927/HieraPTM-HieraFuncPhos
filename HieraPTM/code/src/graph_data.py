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
        if not (root / 'READY.flag').is_file():
            raise FileNotFoundError(f'Graph cache not ready: {root}')
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if not isinstance(manifest, dict):
            raise ValueError('Invalid graph cache manifest')
        self.positions = np.load(root / 'node_positions_1based.int32.npy', mmap_mode='r')
        self.coordinates = np.load(root / 'node_centered_ca.float16.npy', mmap_mode='r')
        self.geometry = np.load(root / 'node_geometry.float16.npy', mmap_mode='r')
        self.masks = np.load(root / 'node_mask.bool.npy', mmap_mode='r')
        with (root / 'sample_index.tsv').open('r', encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        self.index = {row['sample_id']: int(row['row_id']) for row in rows}
        size = len(rows)
        if (len(self.index) != size or self.positions.shape != (size, 33)
                or self.coordinates.shape != (size, 33, 3)
                or self.geometry.shape != (size, 33, 8)
                or self.masks.shape != (size, 33)):
            raise ValueError('Graph cache shape or index mismatch')

    def get_batch(self, sample_ids: list[str]):
        index = np.asarray([self.index[x] for x in sample_ids], dtype=np.int64)
        return (np.asarray(self.positions[index], dtype=np.int32),
                np.asarray(self.coordinates[index], dtype=np.float16),
                np.asarray(self.geometry[index], dtype=np.float16),
                np.asarray(self.masks[index], dtype=np.bool_))

    def close(self) -> None:
        for a in (self.positions, self.coordinates, self.geometry, self.masks):
            close_memmap(a)


class FullProteinNodeReader:
    def __init__(self, protein_index: Path, esm_cache: ESMResidueWindowCache) -> None:
        with protein_index.open('r', encoding='utf-8', newline='') as handle:
            self.proteins = {row['protein_id']: {
                'length': int(row['protein_length']),
                'shard': int(row['cache_shard_id']),
                'offset': int(row['protein_shard_offset0'])}
                for row in csv.DictReader(handle, delimiter='\t')}
        self.esm_cache = esm_cache

    def get_batch(self, protein_ids: list[str], positions: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
        output = np.zeros((*positions.shape, 640), dtype=np.float16)
        shards = {}
        for i, protein_id in enumerate(protein_ids):
            valid = np.flatnonzero(mask[i])
            if not len(valid):
                continue
            meta = self.proteins[protein_id]
            absolute_positions = positions[i, valid]
            if np.any(absolute_positions < 1) or np.any(absolute_positions > meta['length']):
                raise IndexError(f'Graph node outside protein: {protein_id}')
            absolute = meta['offset'] + absolute_positions - 1
            shards.setdefault(meta['shard'], []).append((i, valid, absolute))
        for shard in sorted(shards):
            data = self.esm_cache._mmap(shard)
            for i, valid, absolute in shards[shard]:
                output[i, valid] = np.asarray(data[absolute], dtype=np.float16)
        return output


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def model_logits(model: Any, batch: dict[str, Any]) -> torch.Tensor:
    return model(batch['token_ids'], batch['attention_mask'], batch['esm_embeddings'],
                 batch['esm_present_mask'], batch['node_esm'], batch['node_positions'],
                 batch['node_coordinates'], batch['node_geometry'], batch['node_mask']).reshape(-1)


def evaluate(model: Any, stream: DataLoader, device: torch.device,
             pos_weight: torch.Tensor):
    model.eval()
    labels, scores = [], []
    weighted_loss, rows = 0.0, 0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for raw in stream:
            batch = move(raw, device)
            pred = model_logits(model, batch)
            target = batch['label'].float().reshape(-1)
            weighted_loss += float(criterion(pred, target).item())
            rows += len(target)
            labels.append(target.cpu().numpy())
            scores.append(torch.sigmoid(pred).float().cpu().numpy())
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    y, scores = np.concatenate(labels), np.concatenate(scores)
    metrics = shared.binary_metrics(y, scores)
    metrics.update({'loss': weighted_loss / rows, 'seconds': elapsed,
                    'samples_per_second': rows / max(elapsed, 1e-8),
                    'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == 'cuda' else 0.0})
    return metrics, y, scores, {}
