"""Explicit public replicate mapping, source weights, and runtime options."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .utils import read, sha


TASK1_SHA256 = {
    "ST": "2ebf4e2a3f53f96c0cf080388b2393d3b7251cd86496882b51ac0dc31bd1073d",
    "Y": "aa1581ba10becb27beebeb0358ae6a34ab68340ca14627159f6abc8bc40b8a30",
}


def load(path, replicate):
    path = Path(path).resolve()
    root = path.parent
    config = read(path)
    manifest = read(root / "manifest.json")
    mapping = {item["replicate"]: item["source_replicate"] for item in manifest["source_replicate_mapping"]}
    if not 1 <= replicate <= 9 or mapping.get(replicate) != replicate + 1:
        raise ValueError("Public Rep01–Rep09 must map to source replicates 02–10")
    source_rep = mapping[replicate]
    runtime = config["runtime"]
    if runtime["amp_dtype"] not in ("bfloat16", "float32"):
        raise ValueError("The final pipeline supports the source BF16 or explicit FP32 execution")
    if runtime["workers"] < 0:
        raise ValueError("workers must be nonnegative")
    if runtime["threads"] < 1:
        raise ValueError("threads must be positive")
    lora = SimpleNamespace(**runtime, **config["lora"], seed=20401213 + (source_rep - 1) * 100003)
    dynamic = SimpleNamespace(**runtime, **config["dynamics"],
                              base_seed=20421017 + (source_rep - 1) * 100003,
                              no_amp=runtime["amp_dtype"] == "float32")
    if lora.epochs <= lora.warmup_epochs or lora.warmup_epochs < 0:
        raise ValueError("LoRA epochs must include post-warmup checkpoint selection")
    if dynamic.monitor != "auroc":
        raise ValueError("The final model is selected by validation AUROC")
    for options in (lora, dynamic):
        if options.batch_size < 1 or options.patience < 1 or options.epochs < 1:
            raise ValueError("batch_size, patience, and epochs must be positive")
    weights = {task: (root / name).resolve() for task, name in config["task1_weights"].items()}
    if set(weights) != {"ST", "Y"}:
        raise ValueError("Both matched ST and Y Task1 checkpoints are required")
    for task, filename in weights.items():
        if sha(filename) != TASK1_SHA256[task]:
            raise RuntimeError(f"Not the frozen final Task1 {task} checkpoint: {filename}")
    return SimpleNamespace(root=root, raw=config, lora=lora, dynamic=dynamic, weights=weights,
                           replicate=replicate, source_replicate=source_rep,
                           splits=(root / config["splits"]).resolve(),
                           cache=(root / config["feature_cache_pattern"].format(rep=replicate)).resolve(),
                           run=(root / config["output"] / f"rep{replicate:02d}").resolve())
