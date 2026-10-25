"""Numerical components extracted from the frozen final Task2 snapshot."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from .metrics import metrics
from .utils import move

def optimizer_groups(model, args):
    lora, gates = [], []
    for branch in (model.st, model.y):
        lora.extend(p for name, p in branch.encoder.parent.named_parameters()
                    if p.requires_grad and not name.startswith("gate."))
        gates.extend(branch.encoder.parent.gate.parameters())
        gates.append(branch.encoder.cross_gamma)
    shared = list(model.global_projection.parameters()) + list(model.global_gate.parameters())
    shared += list(model.functional_adapter.parameters()) + list(model.functional_head.parameters())
    all_params = lora + gates + shared
    if len({id(p) for p in all_params}) != len(all_params):
        raise RuntimeError("Duplicate optimizer parameters")
    return [
        {"params": lora, "lr": args.lora_lr},
        {"params": gates, "lr": args.fusion_lr},
        {"params": shared, "lr": args.head_lr},
    ]


def set_warmup(model, warmup: bool):
    for branch in (model.st, model.y):
        for name, parameter in branch.encoder.parent.named_parameters():
            if (name.startswith("gate.") or name.endswith(
                (".q_a.weight", ".q_b.weight", ".v_a.weight", ".v_b.weight",
                 ".a.weight", ".b.weight")
            )) and parameter.requires_grad != (not warmup):
                parameter.requires_grad_(not warmup)
        branch.encoder.cross_gamma.requires_grad_(not warmup)

def epoch(model,loader,args,device,optimizer=None):
    model.train(optimizer is not None); records=[];total=0.;count=0
    keys=('token_ids','attention_mask','esm_embeddings','esm_present_mask','node_esm','node_positions',
          'node_coordinates','node_geometry','node_mask','global_embedding','residue_id')
    for raw in loader:
        batch=move(raw,device)
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                                enabled=optimizer is not None and device.type=='cuda' and args.amp_dtype=='bfloat16'):
                output=model(*[batch[k] for k in keys]);logit=output['logits'].float()
                loss=nn.functional.binary_cross_entropy_with_logits(logit,batch['label'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite base loss; no update applied')
            if optimizer is not None:
                loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step()
        probability=torch.sigmoid(logit.detach()).cpu().numpy()
        for i,site in enumerate(raw['site_id']):
            records.append({'site_id':site,'mmf_label':int(raw['label'][i]),
                            'residue_id':int(raw['residue_id'][i]),'probability':float(probability[i])})
        total+=float(loss.detach())*len(probability);count+=len(probability)
    frame=pd.DataFrame(records)
    return total/count,metrics(frame.mmf_label,frame.probability),frame
