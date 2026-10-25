"""Numerical components extracted from the frozen final Task2 snapshot."""
from __future__ import annotations
import torch
from .functional import HieraPTMFullJointLoRAV2

class InstrumentedBase(HieraPTMFullJointLoRAV2):
    """Same checkpoint tensors/forward as LoRA-v2, plus existing aligned states."""
    def forward(self, token_ids, attention_mask, esm_embeddings, esm_present_mask,
                node_esm, node_positions, node_coordinates, node_geometry, node_mask,
                global_embedding, residue_id):
        batch = len(residue_id)
        representation = esm_embeddings.new_empty((batch,192),dtype=torch.float32)
        static_delta = torch.zeros_like(representation)
        for selected, branch in ((residue_id!=2,self.st),(residue_id==2,self.y)):
            ids = torch.nonzero(selected,as_tuple=False).flatten()
            if not ids.numel():
                continue
            xs = (token_ids,attention_mask,esm_embeddings,esm_present_mask,
                  node_esm,node_positions,node_coordinates,node_geometry,node_mask)
            encoded = branch(*[x.index_select(0,ids) for x in xs])
            representation = representation.index_copy(0,ids,encoded["center_token"].float())
            static_delta = static_delta.index_copy(0,ids,
                (encoded["center_token"].float()-encoded["sequence_center_token"].float()))
        global_vector = self.global_projection(global_embedding.float())
        weight = torch.sigmoid(self.global_gate(torch.cat([representation,global_vector],-1)))
        fused = representation + weight*global_vector
        functional = fused+self.functional_adapter(fused)
        return {"logits":self.functional_head(functional).squeeze(-1),
                "functional":functional,"static_delta":static_delta}


def base_forward(base,batch):
    keys = ("token_ids","attention_mask","esm_embeddings","esm_present_mask","node_esm",
            "node_positions","node_coordinates","node_geometry","node_mask","global_embedding","residue_id")
    return base(*[batch[k] for k in keys])
