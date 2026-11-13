"""Structure-aware encoder for pretrained HieraPTM graph representations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .sequence_fusion import HieraPTMSequenceEncoder
from .structure_graph import DenseStructureGATLayer


class HieraPTMGraphEncoder(nn.Module):
    """Integrate residue-level sequence states with a structural neighborhood."""

    def __init__(self) -> None:
        super().__init__()
        self.variant = "G4"
        self.model_name = "HieraPTM_graph_encoder"

        self.parent = HieraPTMSequenceEncoder()
        self.geometry_adapter = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, 192),
        )
        self.node_input_norm = nn.LayerNorm(192)
        self.graph_layers = nn.ModuleList(
            [DenseStructureGATLayer() for _ in range(2)]
        )
        self.graph_final_norm = nn.LayerNorm(192)

        # Preserve the pretrained checkpoint's pooling and fusion parameters.
        self.graph_pool_score = nn.Sequential(
            nn.Linear(192, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.graph_summary = nn.Sequential(
            nn.Linear(384, 192),
            nn.GELU(),
            nn.LayerNorm(192),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * 192 + 1, 192),
            nn.GELU(),
            nn.Linear(192, 192),
        )
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.constant_(self.fusion_gate[-1].bias, -2.0)
        self.fusion_gamma = nn.Parameter(torch.zeros(()), requires_grad=False)

        for module in (self.fusion_gate, self.graph_pool_score, self.graph_summary):
            module.requires_grad_(False)

        self.cross_query_norm = nn.LayerNorm(192)
        self.cross_graph_norm = nn.LayerNorm(192)
        self.cross_attention = nn.MultiheadAttention(
            192, 4, dropout=0.1, batch_first=True
        )
        self.cross_output_norm = nn.LayerNorm(192)
        self.cross_gamma = nn.Parameter(torch.zeros(()))

        self.freeze_parent()

    def load_parent_state(self, state: dict[str, torch.Tensor]) -> None:
        self.parent.load_state_dict(state, strict=True)
        self.freeze_parent()

    def freeze_parent(self) -> None:
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()

    def train(self, mode: bool = True) -> "HieraPTMGraphEncoder":
        super().train(mode)
        self.parent.eval()
        return self

    def _pool(
        self,
        graph: torch.Tensor,
        h_sequence: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.graph_pool_score(graph).squeeze(-1)
        logits = logits.masked_fill(
            ~node_mask, torch.finfo(logits.dtype).min
        )
        weights = torch.softmax(logits, dim=-1) * node_mask.to(logits.dtype)
        return (weights.unsqueeze(-1) * graph).sum(1), weights

    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor,
        esm_present_mask: torch.Tensor,
        node_esm: torch.Tensor,
        node_positions: torch.Tensor,
        node_coordinates: torch.Tensor,
        node_geometry: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, Any]:
        sequence = self.parent.encode(
            token_ids, attention_mask, esm_embeddings, esm_present_mask
        )
        sequence_tokens = sequence["transformer_tokens"]
        h_sequence = sequence["center_token"]
        node_mask = node_mask.bool()
        graph_present = node_mask[:, 0]

        language = self.parent.esm_projector(node_esm, node_mask)
        graph = self.node_input_norm(
            language + self.geometry_adapter(node_geometry.float())
        )
        graph = graph * node_mask.unsqueeze(-1).to(graph.dtype)

        for layer in self.graph_layers:
            graph = layer(
                graph,
                node_coordinates,
                node_positions,
                node_geometry,
                node_mask,
            )
        graph = self.graph_final_norm(graph)
        graph = graph * node_mask.unsqueeze(-1).to(graph.dtype)

        pooled, pool_weights = self._pool(graph, h_sequence, node_mask)
        h_graph = self.graph_summary(
            torch.cat([graph[:, 0], pooled], dim=-1)
        )
        h_graph = h_graph * graph_present.unsqueeze(-1).to(h_graph.dtype)

        missing = (~graph_present).unsqueeze(-1).to(h_sequence.dtype)
        gate_input = torch.cat(
            [h_sequence, h_graph, h_sequence * h_graph, missing],
            dim=-1,
        )
        gate = torch.sigmoid(self.fusion_gate(gate_input))
        gate = gate * graph_present.unsqueeze(-1).to(h_sequence.dtype)

        safe_mask = node_mask.clone()
        safe_mask[~graph_present, 0] = True
        cross, cross_weights = self.cross_attention(
            self.cross_query_norm(sequence_tokens),
            self.cross_graph_norm(graph),
            self.cross_graph_norm(graph),
            key_padding_mask=~safe_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        cross = self.cross_output_norm(cross)
        cross = cross * graph_present[:, None, None].to(cross.dtype)
        h_final = h_sequence + self.cross_gamma.to(h_sequence.dtype) * cross[:, 30]

        return {
            "center_token": h_final,
            "sequence_center_token": h_sequence,
            "parent_bigru_projected_tokens": sequence["bigru_projected_tokens"],
            "parent_fused_tokens": sequence["fused_tokens"],
            "parent_transformer_tokens": sequence["transformer_tokens"],
            "parent_esm_gate_center_mean": sequence["esm_gate_center_mean"],
            "parent_esm_gate_local_mean": sequence["esm_gate_local_mean"],
            "graph_center_token": graph[:, 0],
            "graph_summary": h_graph,
            "graph_nodes": graph,
            "graph_pool_weights": pool_weights,
            "fusion_gate": gate,
            "graph_present": graph_present,
            "layer_center_attentions": [
                layer.last_center_attention for layer in self.graph_layers
            ],
            "cross_attention_weights": cross_weights,
        }

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor,
        esm_present_mask: torch.Tensor,
        node_esm: torch.Tensor,
        node_positions: torch.Tensor,
        node_coordinates: torch.Tensor,
        node_geometry: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encode(
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
        return self.parent.backbone.classifier(representation["center_token"])[:, 0]

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_graph_variant(variant: str) -> HieraPTMGraphEncoder:
    if variant.upper() != "G4":
        raise ValueError("Only the pretrained G4 graph architecture is supported")
    return HieraPTMGraphEncoder()
