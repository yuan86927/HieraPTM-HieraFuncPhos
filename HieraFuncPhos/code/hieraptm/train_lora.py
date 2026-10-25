"""Stage 1: one functional LoRA reader, selected on fixed validation AUROC."""
from __future__ import annotations

import time

import numpy as np
import torch

from .data import make_loader
from .lora_math import epoch, optimizer_groups, set_warmup
from .models.functional import HieraPTMFullJointLoRAV2
from .utils import save, save_checkpoint, seed_all, sha


def fit(cache, args, weights, contract, directory, device):
    seed_all(args.seed)
    model = HieraPTMFullJointLoRAV2(weights["ST"], weights["Y"], dropout=args.dropout,
                                  lora_dropout=args.lora_dropout, head_width=args.head_width).to(device)
    model.parameter_audit()
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    optimizer = torch.optim.AdamW(optimizer_groups(model, args), lr=args.head_lr, weight_decay=args.weight_decay)
    training = make_loader(cache, cache.roles["train"], args, True, args.seed)
    validation = make_loader(cache, cache.roles["validation"], args, False, args.seed)
    directory.mkdir(parents=True, exist_ok=False)
    best, stale, history = -np.inf, 0, []
    for number in range(1, args.epochs + 1):
        begun = time.monotonic()
        set_warmup(model, number <= args.warmup_epochs)
        tl, tm, _ = epoch(model, training, args, device, optimizer)
        vl, vm, prediction = epoch(model, validation, args, device)
        row = {"epoch": number, "train_loss": tl, "validation_loss": vl,
               "train_auroc": tm["auroc"], "validation_auroc": vm["auroc"],
               "validation_aupr": vm["aupr"], "seconds": time.monotonic() - begun}
        history.append(row)
        save(directory / "history.json", history)
        print({"stage": "functional_lora", **row}, flush=True)
        if number <= args.warmup_epochs:
            continue
        if vm["auroc"] > best + 1e-8:
            best, stale = vm["auroc"], 0
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                     if name in trainable_names}
            save_checkpoint(directory / "best.pt", {
                "contract": contract, "selected_epoch": number, "adapter_state": state,
                "trainable_names": sorted(trainable_names), "model_config": {
                    "dropout": args.dropout, "lora_dropout": args.lora_dropout, "head_width": args.head_width},
                "validation_metrics": vm})
            prediction.to_parquet(directory / "validation_predictions.parquet", index=False)
        else:
            stale += 1
        if stale >= args.patience:
            break
    if not (directory / "best.pt").is_file():
        raise RuntimeError("No post-warmup functional checkpoint was selected")
    save(directory / "complete.json", {"contract": contract, "checkpoint_sha256": sha(directory / "best.pt"),
                                       "test_scored": False})
