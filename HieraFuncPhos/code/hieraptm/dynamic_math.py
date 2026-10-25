"""Numerical components extracted from the frozen final Task2 snapshot."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .utils import save

def tensor_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def configuration(args):
    keys=("batch_size","epochs","patience","lr","dropout","weight_decay","max_logit_delta",
          "logit_penalty","repr_penalty","rank_weight","base_seed","no_amp","amp_dtype","monitor")
    return {**{k:getattr(args,k) for k in keys},"base_always_frozen":True,"head_always_frozen":True,
        "adapter_rank":16,"selection":f"validation_{args.monitor}_with_epoch0_baseline",
        "start_all_dynamic_modules_together":True,"physical_threshold_policy":"unchanged v1 priors; audit train coverage"}


def run_model(model,features,anchors,ids,device):
    ix=torch.as_tensor(ids,device=device)
    return model(anchors["functional"][ix],anchors["static_delta"][ix],anchors["reference_logit"][ix],**features.batch(ids,device))


def autocast(args,device,force_fp32=False):
    dtype=torch.bfloat16 if args.amp_dtype=="bfloat16" else torch.float16
    return torch.autocast(device_type=device.type,dtype=dtype,
        enabled=device.type=="cuda" and not args.no_amp and not force_fp32)


def train_batch(model,features,anchors,ids,labels,protein,args,device,optimizer,scaler,context):
    """Retry the SAME members safely; FP16 overflow must not abort a whole run.

    No optimizer update occurs for a nonfinite attempt. After three mixed-
    precision failures, retry with FP32 arithmetic. Persistent FP32 failures
    retain parameter names and member IDs rather than masking NaNs with zeros.
    """
    events=[]
    ix=torch.as_tensor(ids,device=device)
    for attempt in range(4):
        full_precision=attempt==3
        optimizer.zero_grad(set_to_none=True)
        with autocast(args,device,force_fp32=full_precision):
            result=run_model(model,features,anchors,ids,device)
            loss=nn.functional.binary_cross_entropy_with_logits(result["logits"],labels[ix])
            loss=loss+args.logit_penalty*result["logit_delta"].square().mean()+args.repr_penalty*result["delta_h"].square().mean()
            if args.rank_weight:
                valid=(protein[ix,None]==protein[ix][None])&(labels[ix,None]>.5)&(labels[ix][None]<.5)
                gap=result["logits"][:,None]-result["logits"][None]
                if valid.any():
                    loss=loss+args.rank_weight*nn.functional.softplus(-gap[valid]).mean()
        event=None
        if not torch.isfinite(loss):
            event={"reason":"nonfinite_loss","parameters":[]}
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            norm=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients)))
            if not torch.isfinite(norm):
                names=[name for name,p in model.named_parameters()
                       if p.grad is not None and not torch.isfinite(p.grad).all()]
                if names:
                    event={"reason":"nonfinite_gradient","parameters":names[:20]}
                    if scaler.is_enabled():
                        # unscale_ has registered found_inf. update() backs off and
                        # clears optimizer stage, WITHOUT applying any bad update.
                        old_scale=scaler.get_scale()
                        scaler.update()
                        event.update(old_scale=old_scale,new_scale=scaler.get_scale())
                else:
                    # A sum-of-squares can overflow FP32 even for finite grads.
                    # Recompute this diagnostic in FP64 and scale without overflow.
                    precise=torch.linalg.vector_norm(torch.stack([g.double().norm() for g in gradients]))
                    factor=(1./(precise+1e-6)).clamp(max=1.).float()
                    for gradient in gradients:
                        gradient.mul_(factor)
                    norm=precise
        if event is not None:
            event.update(context,attempt=attempt+1,precision="float32" if full_precision else args.amp_dtype,
                         row_ids=np.asarray(ids,dtype=int).tolist())
            events.append(event)
            print(json.dumps({"HIERADYN_LG_V2_NUMERICAL_RETRY":event}),flush=True)
            if full_precision:
                failure_path=Path(context["failure_directory"])/"numerical_failure.json"
                save(failure_path,{"events":events,"config":configuration(args)})
                raise RuntimeError(f"Persistent nonfinite FP32 calculation; diagnostics: {failure_path}")
            continue
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        return result,loss,norm,events
    raise AssertionError("Unreachable numerical retry state")


def predict(model,features,anchors,ids,args,device,details=False):
    model.eval()
    outputs,extra=[],[]
    with torch.inference_mode():
        for start in range(0,len(ids),args.batch_size):
            block=ids[start:start+args.batch_size]
            with autocast(args,device):
                result=run_model(model,features,anchors,block,device)
            outputs.append(torch.sigmoid(result["logits"].float()).cpu().numpy())
            if details:
                valid=features.arrays["mask"][block]
                vw=result["view_weights"].float().cpu().numpy()
                def average_views(start,stop):
                    active=valid[:,start:stop]
                    return ((vw[:,start:stop]*active[...,None]).sum(1)/active.sum(1,keepdims=True).clip(1)).astype(np.float32)
                extra.append({"row_id":block,"coupling_gate":result["coupling_gate"].float().cpu().numpy(),
                    "scale_weights":result["scale_weights"].float().cpu().numpy(),"logit_delta":result["logit_delta"].float().cpu().numpy(),
                    "delta_h_rms":result["delta_h"].float().square().mean(1).sqrt().cpu().numpy(),
                    "head_cache_offset":result["head_cache_offset"].float().cpu().numpy(),
                    "local_view_weights":average_views(0,25),"global_view_weights":average_views(25,73),
                    "eligible":result["allowed"].any((1,2)).cpu().numpy()})
    return np.concatenate(outputs),extra


def gradient_report(model,step,eligible):
    groups={"semantic":model.semantic,"physics_views":model.physics,"local_encoder":model.local_layers,
            "global_reader":model.global_read,"cross_attention":model.cross,"scale_gate":model.scale_gate,
            "adapter_up":model.up,"adapter_down":model.down}
    result={name:float(torch.sqrt(sum((p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None),
                                     torch.tensor(0.,device=model.up.weight.device)))) for name,module in groups.items()}
    return {"step":step,"eligible_sites_batch":int(eligible),"gradient_norms":result}
