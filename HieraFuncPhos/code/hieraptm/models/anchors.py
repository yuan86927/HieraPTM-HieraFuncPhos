from __future__ import annotations

import torch

from .functional import HieraFuncPhosFunctionalModel


class FunctionalAnchorModel(HieraFuncPhosFunctionalModel):
    def forward(
        self,
        token_ids,
        attention_mask,
        esm_embeddings,
        esm_present_mask,
        node_esm,
        node_positions,
        node_coordinates,
        node_geometry,
        node_mask,
        global_embedding,
        residue_id,
    ):
        batch_size = len(residue_id)
        representation = esm_embeddings.new_empty(
            (batch_size, 192), dtype=torch.float32
        )
        static_delta = torch.zeros_like(representation)

        inputs = (
            token_ids,
            attention_mask,
            esm_embeddings,
            esm_present_mask,
            node_esm,
            node_positions,
            node_coordinates,
            node_geometry,
            node_mask,
        )

        for selected, branch in (
            (residue_id != 2, self.st),
            (residue_id == 2, self.y),
        ):
            indices = torch.nonzero(selected, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue

            encoded = branch(
                *[tensor.index_select(0, indices) for tensor in inputs]
            )
            center = encoded["center_token"].float()
            sequence_center = encoded["sequence_center_token"].float()

            representation = representation.index_copy(0, indices, center)
            static_delta = static_delta.index_copy(
                0, indices, center - sequence_center
            )

        global_vector = self.global_projection(global_embedding.float())
        global_weight = torch.sigmoid(
            self.global_gate(
                torch.cat([representation, global_vector], dim=-1)
            )
        )
        fused = representation + global_weight * global_vector
        functional = fused + self.functional_adapter(fused)

        return {
            "logits": self.functional_head(functional).squeeze(-1),
            "functional": functional,
            "static_delta": static_delta,
        }


def base_forward(base, batch):
    keys = (
        "token_ids",
        "attention_mask",
        "esm_embeddings",
        "esm_present_mask",
        "node_esm",
        "node_positions",
        "node_coordinates",
        "node_geometry",
        "node_mask",
        "global_embedding",
        "residue_id",
    )
    return base(*[batch[key] for key in keys])
