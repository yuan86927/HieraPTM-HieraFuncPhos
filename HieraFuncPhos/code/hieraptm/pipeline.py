"""Two-stage single-split training; test evaluation is a separate command."""
from __future__ import annotations

import gc

import torch

from .anchors import extract
from .config import TASK1_SHA256, load
from .data import DynamicFeatures, FeatureCache
from .dynamic_math import tensor_sha
from .train_dynamics import fit as fit_dynamics
from .train_lora import fit as fit_lora
from .utils import code_hashes, read, runtime_versions, save, sha


def contract_for(options, cache):
    return {
        "protocol": "fixed_train_validation_test_v1",
        "replicate": options.replicate, "source_replicate": options.source_replicate,
        "lora_options": vars(options.lora), "dynamics_options": vars(options.dynamic),
        "task1_checkpoint_sha256": TASK1_SHA256, "feature_audit_sha256": cache.audit_sha256,
        "split_sha256": cache.split_hashes, "code_sha256": code_hashes(options.root),
        "runtime_versions": runtime_versions(),
    }


def train(config_path, replicate):
    options = load(config_path, replicate)
    torch.set_num_threads(options.lora.threads)
    if options.run.exists():
        raise FileExistsError(f"Refusing to overwrite a run; select a new output directory in config.json: {options.run}")
    cache = FeatureCache(options.cache, options.splits, replicate)
    contract = contract_for(options, cache)
    device = torch.device(options.lora.device)
    options.run.mkdir(parents=True, exist_ok=False)
    save(options.run / "run_contract.json", contract)
    fit_lora(cache, options.lora, options.weights, contract, options.run / "functional", device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    anchors, head, anchor_source = extract(cache, options.lora, options.weights,
                                          options.run / "functional/best.pt", contract,
                                          ("train", "validation"), device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sources = {"run_contract": contract, **anchor_source}
    save(options.run / "anchor_provenance.json", anchor_source)
    fit_dynamics(cache.index, DynamicFeatures(cache), options.dynamic, cache.roles, anchors, head,
                 sources, device, options.run / "dynamics", replicate)
    save(options.run / "complete.json", {
        "contract": contract, "functional_sha256": sha(options.run / "functional/best.pt"),
        "dynamics_sha256": sha(options.run / "dynamics/best_adapter.pt"),
        "test_scored": False,
        "selected_dynamics_epoch": read(options.run / "dynamics/validation_metrics.json")["selected_epoch"],
        "frozen_head_sha256": tensor_sha(head),
    })
