"""Numerical components extracted from the frozen final Task2 snapshot."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from .feature_names import FEATURE_NAMES

class DynamicsNormalization:
    def normalize(self,train,record=None):
        fit_hash=hashlib.sha256(np.asarray(train,np.int32).tobytes()).hexdigest()
        if record is not None and (record["fit_members_sha256"]!=fit_hash or record["feature_names"]!=FEATURE_NAMES):
            raise RuntimeError("Normalization belongs to other fit members/fields")
        if record is None:
            mean=np.zeros(28,np.float32)
            std=np.ones(28,np.float32)
            for k in range(28):
                view=0 if k<10 else (1 if k<18 else 2)
                active=np.asarray(self.arrays["mask"][train])&np.asarray(self.arrays["view_mask"][train,:,view])
                values=np.asarray(self.arrays["node"][train,:,k])[active]
                if len(values)<10 or not np.isfinite(values).all():
                    raise RuntimeError(f"Insufficient training values for {FEATURE_NAMES[k]}")
                if k not in (2,4,27):  # agreement and explicit binary flags stay interpretable
                    mean[k],std[k]=values.mean(),max(float(values.std()),1e-5)
            record={"mean":mean.tolist(),"std":std.tolist(),"fit_members_sha256":hashlib.sha256(np.asarray(train,np.int32).tobytes()).hexdigest(),
                    "fit_members":len(train),"feature_names":FEATURE_NAMES}
        mean,std=np.array(record["mean"],np.float32),np.array(record["std"],np.float32)
        self.normalized=np.clip((np.asarray(self.arrays["node"])-mean)/std,-6,6).astype(np.float32)
        for left,right,view in ((0,10,0),(10,18,1),(18,28,2)):
            self.normalized[...,left:right]*=self.arrays["view_mask"][...,view,None]
        self.normalized*=self.arrays["mask"][...,None]
        return record

    def batch(self,ids,device):
        tensor=lambda x:torch.as_tensor(np.array(x[ids],copy=True),device=device)
        result={k:tensor(v) for k,v in self.arrays.items() if k!="node"}
        result["raw_node"],result["node"]=tensor(self.arrays["node"]),tensor(self.normalized)
        result["local_esm"],result["remote_esm"]=tensor(self.local_esm),tensor(self.remote_esm)
        result["site_quality"]=tensor(self.site_quality).float()
        result["site_quality"][:,3]=result["view_mask"][...,1].any(1).float()
        return result
