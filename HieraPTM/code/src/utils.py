"""Final Task1 training dependencies, extracted without changing function bodies."""

from __future__ import annotations
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from metrics import binary_metrics, write_curves


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    last_error: OSError | None = None
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.1 * (attempt + 1))
    raise RuntimeError(f"Could not publish {path}") from last_error


def atomic_json(payload: Any, path: Path) -> None:
    atomic_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", path)


def atomic_tsv(
    rows: list[dict[str, Any]], path: Path, fields: tuple[str, ...] | None = None
) -> None:
    if not rows:
        raise ValueError("Cannot write an empty TSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields or tuple(rows[0])),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    torch.cuda.set_rng_state_all([value.detach().cpu() for value in state["torch_cuda"]])


class FusionIndexedDataset(Dataset):
    def __init__(self, data: FrozenESMHeadData, indices: np.ndarray) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        index = int(self.indices[position])
        metadata = self.data.metadata[index]
        return {
            **metadata,
            "token_ids": self.data.token_ids[index],
            "attention_mask": self.data.attention_masks[index],
            "label": self.data.labels[index],
            "fold_id": self.data.fold_ids[index],
        }


class EpochSampler(Sampler[int]):
    def __init__(self, size: int, seed: int, epoch: int) -> None:
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def is_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    return incumbent is None or (candidate["auroc"], candidate["auprc"]) > (
        incumbent["auroc"], incumbent["auprc"]
    )
