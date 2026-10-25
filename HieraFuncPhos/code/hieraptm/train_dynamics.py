"""Numerical components extracted from the frozen final Task2 snapshot."""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import torch
from .models.dynamics import HieraDynLGV2
from .metrics import metrics
from .utils import save, seed_all
from .dynamic_math import (tensor_sha, configuration, train_batch, predict, gradient_report)

def fit(index, features, args, split, anchors, head, hashes, device, directory, replicate):
    seed_all(args.base_seed)
    model=HieraDynLGV2(head,args.dropout,args.max_logit_delta).to(device)
    frozen_head_hash=tensor_sha(model.head.state_dict())
    normalization=features.normalize(split["train"])
    labels=torch.as_tensor(index.label.to_numpy(np.float32),device=device)
    protein=torch.as_tensor(pd.factorize(index.protein_id,sort=True)[0],device=device)
    trainable=[p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad for p in model.head.parameters()):
        raise RuntimeError("Functional head must remain frozen")
    optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler("cuda",init_scale=1024.,
        enabled=device.type=="cuda" and not args.no_amp and args.amp_dtype=="float16")
    vid=split["validation"]
    baseline=torch.sigmoid(anchors["reference_logit"][vid]).cpu().numpy()
    baseline_metric=metrics(index.label.to_numpy()[vid],baseline)
    best,best_epoch,stale=baseline_metric[args.monitor],0,0
    best_validation=dict(baseline_metric)
    directory.mkdir(parents=True,exist_ok=True)
    def snapshot(epoch):
        tmp=directory/"best_adapter.pt.tmp"
        torch.save({"replicate":replicate,"selected_epoch":epoch,"config":configuration(args),
            "source_sha256":hashes,"normalization":normalization,
            "model_state":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},tmp)
        tmp.replace(directory/"best_adapter.pt")
    snapshot(0)
    history=[{"epoch":0,"validation_auroc":baseline_metric["auroc"],
              "validation_aupr":baseline_metric["aupr"],"selection_metric":f"validation_{args.monitor}"}]
    pd.DataFrame({"site_id":index.site_id.to_numpy()[vid],"probability":baseline}).to_parquet(directory/"validation_predictions.parquet",index=False)
    gradient_logs=[]
    numerical_events=[]
    step=0
    for epoch in range(1,args.epochs+1):
        if device.type=="cuda":
            torch.cuda.synchronize()
        begun=time.perf_counter()
        model.train()
        order=np.random.permutation(split["train"])
        total,count,train_y,train_p,gate_sum=0.,0,[],[],0.
        for start in range(0,len(order),args.batch_size):
            ids=order[start:start+args.batch_size]
            ix=torch.as_tensor(ids,device=device)
            result,loss,norm,events=train_batch(model,features,anchors,ids,labels,protein,args,
                device,optimizer,scaler,{"replicate":replicate,"epoch":epoch,"step":step+1,
                                        "failure_directory":str(directory)})
            numerical_events.extend(events)
            step+=1
            if step in (1,10,100):
                grad=gradient_report(model,step,result["allowed"].any((1,2)).sum())
                gradient_logs.append(grad)
                print(json.dumps({"HIERADYN_LG_V2_GRADIENT":{"replicate":replicate,**grad}}),flush=True)
            count+=len(ids)
            total+=float(loss.detach())*len(ids)
            gate_sum+=float(result["coupling_gate"].detach().sum())
            train_y.append(labels[ix].cpu().numpy())
            train_p.append(torch.sigmoid(result["logits"].detach().float()).cpu().numpy())
        val_p,_=predict(model,features,anchors,vid,args,device)
        vm=metrics(index.label.to_numpy()[vid],val_p)
        tm=metrics(np.concatenate(train_y),np.concatenate(train_p))
        if tensor_sha(model.head.state_dict())!=frozen_head_hash:
            raise RuntimeError("Frozen functional head changed")
        row={"epoch":epoch,"phase":"frozen_base_dynamic_joint","train_loss":total/count,
            "train_auroc":tm["auroc"],"train_aupr":tm["aupr"],"validation_auroc":vm["auroc"],
            "validation_aupr":vm["aupr"],"coupling_gate_mean":gate_sum/count,
            "seconds":round(time.perf_counter()-begun,2),"frozen_head_unchanged":True,
            "amp_dtype":args.amp_dtype,"numerical_retries_total":len(numerical_events),
            "selection_metric":f"validation_{args.monitor}"}
        history.append(row)
        print(json.dumps({"HIERADYN_LG_V2_EPOCH":{"replicate":replicate,**row}}),flush=True)
        if vm[args.monitor]>best+1e-4:
            best,best_epoch,stale=vm[args.monitor],epoch,0
            best_validation=dict(vm)
            snapshot(epoch)
            pd.DataFrame({"site_id":index.site_id.to_numpy()[vid],"probability":val_p}).to_parquet(directory/"validation_predictions.parquet",index=False)
        else:
            stale+=1
        save(directory/"history.json",history)
        save(directory/"gradient_audit.json",gradient_logs)
        save(directory/"numerics_audit.json",numerical_events)
        if stale>=args.patience:
            break
    report={"status":"HIERAPTM_SINGLE_SPLIT_PASS","replicate":replicate,"selected_epoch":best_epoch,
        "selection_metric":f"validation_{args.monitor}","best_selection_value":best,
        "best_validation_auroc":best_validation["auroc"],"best_validation_aupr":best_validation["aupr"],
        "baseline_validation":baseline_metric,"source_sha256":hashes,
        "config":configuration(args),"outer_test_scored":False,
        "trainable_auxiliary_parameters":sum(p.numel() for p in trainable),
        "main_model_trainable_parameters":0,"head_unchanged":True}
    save(directory/"validation_metrics.json",report)
    return report
