"""Train the HieraFuncPhos dynamics adapter."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from .dynamic_math import (
    configuration,
    gradient_report,
    predict,
    tensor_sha,
    train_batch,
)
from .metrics import metrics
from .models.dynamics import HieraFuncPhosDynamics
from .utils import save, seed_all


def fit(index, features, args, split, anchors, head, hashes, device, directory, replicate):
    seed_all(args.base_seed)
    model = HieraFuncPhosDynamics(
        head, args.dropout, args.max_logit_delta
    ).to(device)
    frozen_head_hash = tensor_sha(model.head.state_dict())
    normalization = features.normalize(split["train"])

    labels = torch.as_tensor(index.label.to_numpy(np.float32), device=device)
    protein = torch.as_tensor(
        pd.factorize(index.protein_id, sort=True)[0], device=device
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(parameter.requires_grad for parameter in model.head.parameters()):
        raise RuntimeError("Functional prediction head must remain frozen")

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=1024.0,
        enabled=(
            device.type == "cuda"
            and not args.no_amp
            and args.amp_dtype == "float16"
        ),
    )

    validation_ids = split["validation"]
    baseline = torch.sigmoid(
        anchors["reference_logit"][validation_ids]
    ).cpu().numpy()
    baseline_metric = metrics(index.label.to_numpy()[validation_ids], baseline)
    best = baseline_metric[args.monitor]
    best_epoch = 0
    stale = 0
    best_validation = dict(baseline_metric)

    directory.mkdir(parents=True, exist_ok=True)

    def save_selected_checkpoint(epoch):
        temporary = directory / "best_adapter.pt.tmp"
        torch.save(
            {
                "replicate": replicate,
                "selected_epoch": epoch,
                "config": configuration(args),
                "source_sha256": hashes,
                "normalization": normalization,
                "model_state": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            },
            temporary,
        )
        temporary.replace(directory / "best_adapter.pt")

    save_selected_checkpoint(0)
    history = [
        {
            "epoch": 0,
            "validation_auroc": baseline_metric["auroc"],
            "validation_aupr": baseline_metric["aupr"],
            "selection_metric": f"validation_{args.monitor}",
        }
    ]
    pd.DataFrame(
        {
            "site_id": index.site_id.to_numpy()[validation_ids],
            "probability": baseline,
        }
    ).to_parquet(directory / "validation_predictions.parquet", index=False)

    gradient_logs = []
    numerical_events = []
    step = 0

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model.train()
        order = np.random.permutation(split["train"])
        total, count = 0.0, 0
        train_y, train_p = [], []
        gate_sum = 0.0

        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            ix = torch.as_tensor(ids, device=device)
            result, loss, norm, events = train_batch(
                model,
                features,
                anchors,
                ids,
                labels,
                protein,
                args,
                device,
                optimizer,
                scaler,
                {
                    "replicate": replicate,
                    "epoch": epoch,
                    "step": step + 1,
                    "failure_directory": str(directory),
                },
            )
            numerical_events.extend(events)
            step += 1

            if step in (1, 10, 100):
                grad = gradient_report(
                    model, step, result["allowed"].any((1, 2)).sum()
                )
                gradient_logs.append(grad)
                print(
                    json.dumps(
                        {"HieraFuncPhos_gradient": {"replicate": replicate, **grad}}
                    ),
                    flush=True,
                )

            count += len(ids)
            total += float(loss.detach()) * len(ids)
            gate_sum += float(result["coupling_gate"].detach().sum())
            train_y.append(labels[ix].cpu().numpy())
            train_p.append(
                torch.sigmoid(result["logits"].detach().float()).cpu().numpy()
            )

        validation_probabilities, _ = predict(
            model, features, anchors, validation_ids, args, device
        )
        validation_metrics = metrics(
            index.label.to_numpy()[validation_ids], validation_probabilities
        )
        train_metrics = metrics(
            np.concatenate(train_y), np.concatenate(train_p)
        )

        if tensor_sha(model.head.state_dict()) != frozen_head_hash:
            raise RuntimeError("Frozen functional prediction head changed")

        row = {
            "epoch": epoch,
            "phase": "dynamics_training",
            "train_loss": total / count,
            "train_auroc": train_metrics["auroc"],
            "train_aupr": train_metrics["aupr"],
            "validation_auroc": validation_metrics["auroc"],
            "validation_aupr": validation_metrics["aupr"],
            "coupling_gate_mean": gate_sum / count,
            "seconds": round(time.perf_counter() - started, 2),
            "frozen_head_unchanged": True,
            "amp_dtype": args.amp_dtype,
            "numerical_retries_total": len(numerical_events),
            "selection_metric": f"validation_{args.monitor}",
        }
        history.append(row)
        print(
            json.dumps({"HieraFuncPhos_epoch": {"replicate": replicate, **row}}),
            flush=True,
        )

        if validation_metrics[args.monitor] > best + 1e-4:
            best, best_epoch, stale = validation_metrics[args.monitor], epoch, 0
            best_validation = dict(validation_metrics)
            save_selected_checkpoint(epoch)
            pd.DataFrame(
                {
                    "site_id": index.site_id.to_numpy()[validation_ids],
                    "probability": validation_probabilities,
                }
            ).to_parquet(
                directory / "validation_predictions.parquet", index=False
            )
        else:
            stale += 1

        save(directory / "history.json", history)
        save(directory / "gradient_audit.json", gradient_logs)
        save(directory / "numerics_audit.json", numerical_events)
        if stale >= args.patience:
            break

    report = {
        "status": "HIERAFUNCPHOS_TRAINING_COMPLETE",
        "replicate": replicate,
        "selected_epoch": best_epoch,
        "selection_metric": f"validation_{args.monitor}",
        "best_selection_value": best,
        "best_validation_auroc": best_validation["auroc"],
        "best_validation_aupr": best_validation["aupr"],
        "baseline_validation": baseline_metric,
        "source_sha256": hashes,
        "config": configuration(args),
        "outer_test_scored": False,
        "trainable_dynamics_parameters": sum(p.numel() for p in trainable),
        "functional_model_trainable_parameters": 0,
        "head_unchanged": True,
    }
    save(directory / "validation_metrics.json", report)
    return report
