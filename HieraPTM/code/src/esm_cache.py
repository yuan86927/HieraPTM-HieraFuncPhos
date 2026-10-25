"""Read-only random access to frozen ESM-2 residue shards as padded 61-token windows."""

from __future__ import annotations

import csv
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def close_memmap(array: np.ndarray) -> None:
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        memory_map.close()


class ESMResidueWindowCache:
    """Lazy, read-only LRU mmap reader for fixed 61-residue site windows."""

    def __init__(
        self,
        cache_root: str | Path,
        registry_path: str | Path,
        index_path: str | Path,
        *,
        window_length: int = 61,
        dimension: int = 640,
        maximum_open_shards: int = 4,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.window_length = int(window_length)
        self.dimension = int(dimension)
        self.maximum_open_shards = int(maximum_open_shards)
        if self.maximum_open_shards < 1:
            raise ValueError("maximum_open_shards must be positive")

        registry_rows = read_tsv(Path(registry_path))
        self.shards: dict[int, dict] = {}
        for row in registry_rows:
            shard_id = int(row["cache_shard_id"])
            path = self.cache_root / row["array_relative_path"]
            self.shards[shard_id] = {
                "path": path,
                "rows": int(row["unique_residues"]),
                "sha256": row["array_sha256"],
                "status": row["status"],
            }
        if not self.shards or any(row["status"] != "COMPLETE" for row in self.shards.values()):
            raise ValueError("Cache registry contains missing or incomplete shards")

        self.index: dict[str, dict] = {}
        for row in read_tsv(Path(index_path)):
            sample_id = row["sample_id"]
            if sample_id in self.index:
                raise ValueError(f"Duplicate sample_id in cache index: {sample_id}")
            self.index[sample_id] = {
                "sample_id": sample_id,
                "protein_id": row["protein_id"],
                "center_position_1based": int(row["center_position_1based"]),
                "cache_shard_id": int(row["cache_shard_id"]),
                "protein_shard_offset0": int(row["protein_shard_offset0"]),
                "window_protein_start0": int(row["window_protein_start0"]),
                "window_protein_end0_exclusive": int(row["window_protein_end0_exclusive"]),
                "window_shard_start0": int(row["window_shard_start0"]),
                "window_shard_end0_exclusive": int(row["window_shard_end0_exclusive"]),
                "left_padding": int(row["left_padding"]),
                "right_padding": int(row["right_padding"]),
                "valid_mask_61": row["valid_mask_61"],
            }
        self._open: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def __enter__(self) -> "ESMResidueWindowCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for array in self._open.values():
            close_memmap(array)
        self._open.clear()

    @property
    def open_shard_count(self) -> int:
        return len(self._open)

    def _mmap(self, shard_id: int) -> np.ndarray:
        if shard_id not in self.shards:
            raise KeyError(f"Unknown cache shard: {shard_id}")
        if shard_id in self._open:
            array = self._open.pop(shard_id)
            self._open[shard_id] = array
            return array
        metadata = self.shards[shard_id]
        # Evict before opening the next ~610 MiB mapping.  Evicting afterwards
        # briefly requires maximum_open_shards + 1 mappings and can trigger
        # WinError 1455 on Windows systems with a small paging file.
        while len(self._open) >= self.maximum_open_shards:
            _, evicted = self._open.popitem(last=False)
            close_memmap(evicted)
        array = np.load(metadata["path"], mmap_mode="r")
        expected_shape = (metadata["rows"], self.dimension)
        if array.shape != expected_shape or array.dtype != np.float16:
            close_memmap(array)
            raise ValueError(f"Shard {shard_id} has shape/dtype {array.shape}/{array.dtype}, expected {expected_shape}/float16")
        self._open[shard_id] = array
        return array

    def metadata(self, sample_id: str) -> dict:
        try:
            return self.index[sample_id]
        except KeyError as error:
            raise KeyError(f"Unknown sample_id: {sample_id}") from error

    def _read_into(self, sample_id: str, output: np.ndarray, mask: np.ndarray) -> None:
        row = self.metadata(sample_id)
        left = row["left_padding"]
        right = row["right_padding"]
        source_start = row["window_shard_start0"]
        source_end = row["window_shard_end0_exclusive"]
        destination_end = self.window_length - right
        if left < 0 or right < 0 or destination_end < left:
            raise ValueError(f"Invalid padding for {sample_id}: {left}/{right}")
        if source_end - source_start != destination_end - left:
            raise ValueError(f"Window/cache length mismatch for {sample_id}")
        shard_rows = self.shards[row["cache_shard_id"]]["rows"]
        if source_start < 0 or source_end > shard_rows:
            raise IndexError(f"Cache slice out of bounds for {sample_id}")
        expected_mask = np.fromiter((character == "1" for character in row["valid_mask_61"]), dtype=np.bool_, count=self.window_length)
        if len(row["valid_mask_61"]) != self.window_length:
            raise ValueError(f"Mask length is not {self.window_length} for {sample_id}")
        derived_mask = np.zeros(self.window_length, dtype=np.bool_)
        derived_mask[left:destination_end] = True
        if not np.array_equal(expected_mask, derived_mask):
            raise ValueError(f"Stored/derived mask mismatch for {sample_id}")
        source = self._mmap(row["cache_shard_id"])[source_start:source_end]
        output[left:destination_end] = source
        mask[:] = expected_mask

    def get(self, sample_id: str, *, output_dtype: np.dtype = np.float32) -> tuple[np.ndarray, np.ndarray]:
        output = np.zeros((self.window_length, self.dimension), dtype=output_dtype)
        mask = np.zeros(self.window_length, dtype=np.bool_)
        self._read_into(sample_id, output, mask)
        return output, mask

    def get_batch(
        self,
        sample_ids: Sequence[str] | Iterable[str],
        *,
        output_dtype: np.dtype = np.float32,
    ) -> tuple[np.ndarray, np.ndarray]:
        identifiers = list(sample_ids)
        output = np.zeros((len(identifiers), self.window_length, self.dimension), dtype=output_dtype)
        masks = np.zeros((len(identifiers), self.window_length), dtype=np.bool_)
        positions_by_shard: dict[int, list[int]] = defaultdict(list)
        for position, sample_id in enumerate(identifiers):
            positions_by_shard[self.metadata(sample_id)["cache_shard_id"]].append(position)
        for shard_id in sorted(positions_by_shard):
            self._mmap(shard_id)
            for position in positions_by_shard[shard_id]:
                self._read_into(identifiers[position], output[position], masks[position])
        return output, masks
