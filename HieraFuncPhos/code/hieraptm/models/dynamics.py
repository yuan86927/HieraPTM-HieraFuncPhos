"""Frozen LoRA-v2 head + MRN Local/Global conditional representation adapter."""
from __future__ import annotations
import math
import torch
from torch import nn
from .dynamic_attention import masked_attention


def masked_softmax(scores,mask,dim=-1):
    weights=torch.softmax(scores.float().masked_fill(~mask,-1e9),dim=dim)*mask
    return weights/weights.sum(dim=dim,keepdim=True).clamp_min(1e-12)


class PhysicsViews(nn.Module):
    def __init__(self,width=128):
        super().__init__()
        self.encoders=nn.ModuleList([nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,width)) for d in (10,8,10)])
        self.condition=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,32))
        self.view_embed=nn.Embedding(3,8)
        self.scale_embed=nn.Embedding(2,8)
        self.gate=nn.Sequential(nn.Linear(width+32+4+8+8,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,node,quality,view_mask,functional):
        chunks=(node[...,:10],node[...,10:18],node[...,18:28])
        values=torch.stack([encoder(x) for encoder,x in zip(self.encoders,chunks)],2)
        b,n,_,d=values.shape
        query=self.condition(functional).view(b,1,1,32).expand(-1,n,3,-1)
        scales=torch.cat([torch.zeros(25,dtype=torch.long,device=node.device),
                          torch.ones(48,dtype=torch.long,device=node.device)])
        se=self.scale_embed(scales).view(1,n,1,8).expand(b,-1,3,-1)
        ve=self.view_embed(torch.arange(3,device=node.device)).view(1,1,3,8).expand(b,n,-1,-1)
        q=quality[:,:,None].expand(-1,-1,3,-1)
        scores=self.gate(torch.cat([values,query,q,se,ve],-1))[...,0]
        weights=masked_softmax(scores,view_mask)
        return (values*weights[...,None]).sum(2),weights


class MRNEdgeBias(nn.Module):
    def __init__(self,heads=4):
        super().__init__()
        self.motion=nn.Sequential(nn.Linear(5,24),nn.GELU(),nn.Linear(24,heads))
        self.response=nn.Sequential(nn.Linear(3,24),nn.GELU(),nn.Linear(24,heads))
        self.network=nn.Sequential(nn.Linear(4,24),nn.GELU(),nn.Linear(24,heads))
        self.view_gate=nn.Linear(12,3)
        self.functional_gate=nn.Linear(192,3)
        self.geometry=nn.Linear(2,heads,bias=False)

    def forward(self,left,right,left_valid,right_valid,pair,reverse_prs,functional):
        # On-disk pair/PRS caches are float16. Audit/CPU forwards may run
        # without autocast, so never pass Half directly to FP32 projections.
        left,right=left.float(),right.float()
        pair,reverse_prs,functional=pair.float(),reverse_prs.float(),functional.float()
        b,nq,nk,_=pair.shape
        available=left_valid[:,:,None,:]&right_valid[:,None,:,:]
        anm=left[:,:,None,4]*right[:,None,:,4]
        motion=torch.cat([pair[...,:4],anm[...,None]],-1)
        response=torch.stack([pair[...,4],reverse_prs,available[...,1].float()],-1)
        network=torch.stack([pair[...,7],
            (left[:,:,None,19]+right[:,None,:,19])/2,
            (left[:,:,None,20]-right[:,None,:,20]).abs(),
            (left[:,:,None,24]+right[:,None,:,24])/2],-1)
        motion=motion*available[...,0:1]
        response=response*available[...,1:2]
        network=network*available[...,2:3]
        scores=self.view_gate(torch.cat([motion,response,network],-1))
        scores=scores+self.functional_gate(functional)[:,None,None]
        weights=masked_softmax(scores,available)
        values=torch.stack([self.motion(motion),self.response(response),self.network(network)],-2)
        bias=(weights[...,None]*values).sum(-2)+self.geometry(pair[...,5:7])
        return bias.permute(0,3,1,2),weights


class TokenAttention(nn.Module):
    def __init__(self,width=128,heads=4,dropout=.25):
        super().__init__()
        self.heads=heads
        self.q,self.k,self.v=[nn.Linear(width,width) for _ in range(3)]
        self.qnorm,self.knorm=nn.LayerNorm(width),nn.LayerNorm(width)
        self.output=nn.Linear(width,width)
        self.drop=nn.Dropout(dropout)

    def forward(self,query,source,allowed,bias):
        b,nq,d=query.shape
        nk=source.shape[1]
        reshape=lambda t,n:t.reshape(b,n,self.heads,d//self.heads).transpose(1,2)
        q=reshape(self.q(self.qnorm(query)),nq)
        k=reshape(self.k(self.knorm(source)),nk)
        v=reshape(self.v(self.knorm(source)),nk)
        message,weights=masked_attention(q,k,v,allowed,bias,self.drop)
        message=message.transpose(1,2).reshape(b,nq,d)
        return self.drop(self.output(message))*allowed.any(-1,keepdim=True),weights


class HieraDynLGV2(nn.Module):
    def __init__(self,head_state,dropout=.25,max_logit_delta=.5):
        super().__init__()
        self.max_logit_delta=max_logit_delta
        self.head=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,128),nn.GELU(),
                                 nn.Dropout(.30),nn.Linear(128,1))
        self.head.load_state_dict(head_state,strict=True)
        self.head.requires_grad_(False)
        self.head.eval()
        self.semantic=nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,128),nn.GELU())
        self.physics=PhysicsViews()
        self.token_norm=nn.LayerNorm(128)
        self.edge_bias=MRNEdgeBias()
        self.local_layers=nn.ModuleList([TokenAttention(dropout=dropout) for _ in range(2)])
        self.local_ffn=nn.ModuleList([nn.Sequential(nn.LayerNorm(128),nn.Linear(128,192),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(192,128)) for _ in range(2)])
        self.cross=TokenAttention(dropout=dropout)
        self.cross_gate=nn.Sequential(nn.Linear(192+5,32),nn.GELU(),nn.Linear(32,1))
        nn.init.constant_(self.cross_gate[-1].bias,-2.)
        self.function_queries=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,3*128))
        self.local_read=TokenAttention(dropout=dropout)
        self.global_read=TokenAttention(dropout=dropout)
        self.signed_merge=nn.Sequential(nn.Linear(258,128),nn.GELU(),nn.Dropout(dropout))
        self.scale_gate=nn.Sequential(nn.LayerNorm(192+128+128+5),nn.Linear(453,64),nn.GELU(),nn.Linear(64,3))
        self.adapter_gate=nn.Sequential(nn.LayerNorm(192+192+128+5),nn.Linear(517,64),nn.GELU(),nn.Linear(64,1))
        self.state_norm=nn.LayerNorm(192)
        self.down=nn.Linear(192,16,bias=False)
        self.modulation=nn.Linear(128,16)
        self.up=nn.Linear(16,192,bias=False)
        nn.init.zeros_(self.up.weight)
        self.thresholds={"correlation":.35,"agreement":.75,"std":.20,"plddt":.70,"center_plddt":.50,
                         "anm":.20,"prs":.10,"gnm_only_correlation":.60}

    def train(self,mode=True):
        super().train(mode)
        self.head.eval()
        return self

    def physical_masks(self,raw,quality,view_mask,mask,cross):
        t=self.thresholds
        local,global_=mask[:,:25],mask[:,25:]
        mapped=local[:,:,None]&global_[:,None,:]
        mapped=mapped.clone()
        mapped[:,0]=False
        spectral=view_mask[:,:25,None,0]&view_mask[:,None,25:,0]
        reliable=(quality[:,:25,None,0]>=t["plddt"])&(quality[:,None,25:,0]>=t["plddt"])
        reliable&=quality[:,0:1,None,0]>=t["center_plddt"]
        stable=(cross[...,0].abs()>=t["correlation"])&(cross[...,1]<=t["std"])&(cross[...,2]>=t["agreement"])
        anm=(raw[:,:25,None,4]>.5)&(raw[:,None,25:,4]>.5)
        support=(cross[...,3].abs()>=t["anm"])|(cross[...,4]>=t["prs"])
        gnm_only=(cross[...,0].abs()>=.60)&(cross[...,1]<=.10)&(cross[...,2]>=.999)
        eligible=mapped&spectral&reliable&stable&torch.where(anm,support,gnm_only)
        return {"mapped":mapped,"spectral":mapped&spectral,
                "confidence":mapped&spectral&reliable,"stable":mapped&spectral&reliable&stable,
                "eligible":eligible}

    def forward(self,functional,static_delta,base_logit,remote_esm,local_esm,node,raw_node,
                view_mask,quality,mask,site_quality,local_pair,cross,cross_reverse_prs,center_pair):
        functional,static_delta=functional.float(),static_delta.float()
        node,raw_node,quality=node.float(),raw_node.float(),quality.float()
        site_quality=site_quality.float()
        local_pair,cross=local_pair.float(),cross.float()
        cross_reverse_prs,center_pair=cross_reverse_prs.float(),center_pair.float()
        physics,view_weights=self.physics(node,quality,view_mask,functional)
        semantic=torch.cat([self.semantic(local_esm.float()),self.semantic(remote_esm.float())],1)
        tokens=self.token_norm(semantic+physics)*mask[...,None]
        local,global_=tokens[:,:25],tokens[:,25:]
        lm,gm=mask[:,:25],mask[:,25:]
        raw_l,raw_g=raw_node[:,:25],raw_node[:,25:]
        lv,gv=view_mask[:,:25],view_mask[:,25:]
        local_bias,_=self.edge_bias(raw_l,raw_l,lv,lv,local_pair,local_pair.transpose(1,2)[...,4],functional)
        adjacency=lm[:,:,None]&lm[:,None,:]
        adjacency&=(local_pair[...,5]<=.12)|(local_pair[...,6]<=math.log(2)/10+.001)|(
            (local_pair[...,0].abs()>=.35)&(local_pair[...,2]>=.75))
        for layer,ffn in zip(self.local_layers,self.local_ffn):
            message,_=layer(local,local,adjacency,local_bias)
            local=(local+message)*lm[...,None]
            local=(local+ffn(local))*lm[...,None]
        allowed=self.physical_masks(raw_node,quality,view_mask,mask,cross)["eligible"]
        cross_bias,edge_view_weights=self.edge_bias(raw_l,raw_g,lv,gv,cross,cross_reverse_prs,functional)
        strength=cross[...,0].abs()*cross[...,2]
        avg=(strength*allowed).sum((1,2))/allowed.sum((1,2)).clamp_min(1)
        reliability=torch.stack([avg,allowed.float().mean((1,2)),allowed.any(1).float().mean(1),
            quality[:,0,0],gv[...,1].float().mean(1)],-1)
        cg=torch.sigmoid(self.cross_gate(torch.cat([functional,reliability],-1)))[:,0]*allowed.any((1,2))
        update,_=self.cross(global_,local,allowed.transpose(1,2),cross_bias.transpose(2,3))
        global_=global_+cg[:,None,None]*update
        queries=self.function_queries(functional).reshape(len(functional),3,128)
        zero_bias=local.new_zeros((len(local),4,1,25))
        local_value,_=self.local_read(queries[:,:1],local,lm[:,None],zero_bias)
        local_value=local_value[:,0]
        center_bias,_=self.edge_bias(raw_node[:,:1],raw_g,view_mask[:,:1],gv,
            center_pair[:,None],raw_g[...,11][:,None],functional)
        global_values,attentions,sign_valid=[],[],[]
        for idx,sign in enumerate((1,-1),1):
            signed=gm&(raw_g[...,0]*sign>0)
            value,attention=self.global_read(queries[:,idx:idx+1],global_,signed[:,None],center_bias)
            global_values.append(value[:,0])
            attentions.append(attention[:,:,0])
            sign_valid.append(signed.any(1))
        signed_valid=torch.stack(sign_valid,-1)
        global_value=self.signed_merge(torch.cat([*global_values,signed_valid.float()],-1))*signed_valid.any(-1,keepdim=True)
        valid_l,valid_g=lm[:,1:].any(1),gm.any(1)
        valid_scales=torch.stack([torch.ones_like(valid_l),valid_l,valid_g],-1)
        weights=masked_softmax(self.scale_gate(torch.cat([functional,local_value,global_value,site_quality],-1)),valid_scales)
        dynamics=weights[:,1:2]*local_value+weights[:,2:3]*global_value
        gate=torch.sigmoid(self.adapter_gate(torch.cat([functional,static_delta,dynamics,site_quality],-1)))
        use=(valid_l|valid_g).float()[:,None]*(1-weights[:,:1])
        rank=self.down(self.state_norm(functional))*torch.sigmoid(self.modulation(dynamics))
        delta=use*gate*self.up(rank)
        # Evaluate both heads in identical FP32 arithmetic. A fixed cache/AMP
        # offset must not become a learned "dynamic" gain at initialization.
        with torch.autocast(device_type=functional.device.type,enabled=False):
            original=self.head(functional.float()).squeeze(-1)
            adapted=self.head(functional.float()+delta.float()).squeeze(-1)
            difference=adapted-original
            bounded=self.max_logit_delta*torch.tanh(difference/self.max_logit_delta)
        bounded=bounded*(valid_l|valid_g)
        return {"logits":base_logit.float()+bounded,"logit_delta":bounded,"delta_h":delta,
                "gate":gate[:,0]*(valid_l|valid_g),"scale_weights":weights,
                "coupling_gate":cg,"allowed":allowed,"view_weights":view_weights,
                "edge_view_weights":edge_view_weights,"signed_attention":torch.stack(attentions,1),
                "head_cache_offset":base_logit.float()-original.detach()}
