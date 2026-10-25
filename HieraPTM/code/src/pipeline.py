"""Shared resource setup and dispatch for the separate training/test commands."""

from __future__ import annotations

import copy
import gc
import json
import platform
from contextlib import contextmanager

import numpy as np
import torch

import graph_data as graph
import utils as shared
from configuration import resolve
from data import SafeFullProteinNodeReader, checked_data, feature_contract
from esm_cache import ESMResidueWindowCache


@contextmanager
def open_features(config: dict):
    paths = config["paths"]
    cache = ESMResidueWindowCache(resolve(paths["esm_cache_root"]), resolve(paths["esm_cache_registry"]),
                                  resolve(paths["prepared_site_index"]), maximum_open_shards=2)
    try:
        graph_cache = graph.GraphCache(resolve(paths["graph_cache_root"]))
        try:
            reader = SafeFullProteinNodeReader(resolve(paths["prepared_protein_index"]), cache)
            yield cache, graph_cache, reader
        finally:
            graph_cache.close()
    finally:
        cache.close()


def require_selected_models(config: dict, heads: list[str]) -> None:
    output = resolve(config["paths"]["output_root"])
    for head in heads:
        for relative in ("sequence_backbone/COMPLETE.flag", "sequence_backbone/checkpoints/best.pt",
                         "HieraPTM/SCREEN_COMPLETE.flag", "HieraPTM/screen_metrics.json", "HieraPTM/checkpoints/best.pt"):
            if not (output / head / relative).is_file():
                raise FileNotFoundError(f"Train/select every head first: {output / head / relative}")


def run(config: dict, heads: list[str], stage: str) -> None:
    if stage not in ("train", "evaluate", "check-data", "check-inputs"):
        raise ValueError(f"Unknown stage: {stage}")
    if stage in ("check-data", "check-inputs"):
        for head in heads:
            data, identity = checked_data(config, head)
            print(json.dumps({"head": head, "counts": data.counts(), **identity}), flush=True)
            del data
            gc.collect()
        if stage == "check-data":
            return
    if stage == "evaluate":
        require_selected_models(config, heads)
    if stage in ("train", "evaluate") and not torch.cuda.is_available():
        raise RuntimeError("Full training/evaluation requires a CUDA GPU")
    feature_identity = feature_contract(config)
    with open_features(config) as (cache, graph_cache, reader):
        if stage == "check-inputs":
            absent = [str(meta["path"]) for meta in cache.shards.values() if not meta["path"].is_file()]
            if absent:
                raise FileNotFoundError("Missing ESM shards: " + ", ".join(absent))
            print(json.dumps({"status": "INPUT_PATHS_AND_DATA_PASS", "heads": heads,
                              "esm_shards": len(cache.shards), **feature_identity}), flush=True)
            return
        output = resolve(config["paths"]["output_root"])
        runtime = {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0)}
        shared.atomic_json(runtime, output / f"runtime_{stage}.json")
        for head in heads:
            data, split_identity = checked_data(config, head)
            current = copy.deepcopy(config)
            current["protocol"]["development_seed"] = current["protocol"]["development_seed_by_head"][head]
            current["_config_sha256"] = shared.canonical_sha256(config)
            identity = {**feature_identity, **split_identity}
            shared.atomic_json(current, output / head / "resolved_config.json")
            if stage == "train":
                from training import train_sequence_backbone, train_hieraptm
                train_sequence_backbone(current, head, data, cache, torch.device("cuda:0"), identity)
                train_hieraptm(current, head, data, cache, graph_cache, reader, torch.device("cuda:0"), identity)
            else:
                from evaluation import evaluate_head
                evaluate_head(current, head, data, cache, graph_cache, reader, torch.device("cuda:0"), identity)
            del data
            gc.collect()
            torch.cuda.empty_cache()
