"""Independent whole-protein structure Graph encoder with late SequenceBackbone fusion."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sequence_backbone import SequenceBackbone


class DenseStructureGATLayer(nn.Module):
    def __init__(self, d_model: int = 192, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads; self.head_dim = d_model // heads; self.scale = self.head_dim ** -0.5
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.edge_mlp = nn.Sequential(nn.Linear(10, 16), nn.GELU(), nn.Linear(16, heads))
        self.output = nn.Linear(d_model, d_model); self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 384), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(384, d_model), nn.Dropout(dropout))
        self.register_buffer("rbf_centers", torch.linspace(2.0, 12.0, 8), persistent=True)
        self.last_center_attention: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor, coordinates: torch.Tensor, positions: torch.Tensor,
                geometry: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        batch, nodes, width = hidden.shape
        normalized = self.input_norm(hidden)
        qkv = self.qkv(normalized).view(batch, nodes, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        scores = torch.einsum("bnhd,bmhd->bhnm", query, key) * self.scale
        distance = torch.cdist(coordinates.float(), coordinates.float()).clamp(0, 32)
        sequence_adjacent = (positions[:, :, None] - positions[:, None, :]).abs().eq(1)
        eye = torch.eye(nodes, dtype=torch.bool, device=hidden.device).view(1, nodes, nodes)
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        adjacency = valid & (distance.le(8.0) | sequence_adjacent | eye)
        rbf = torch.exp(-0.5 * ((distance.unsqueeze(-1) - self.rbf_centers) / 1.5).square())
        confidence = torch.sqrt(
            geometry[:, :, 1].float().clamp(0, 1).unsqueeze(2)
            * geometry[:, :, 1].float().clamp(0, 1).unsqueeze(1)
        )
        edge = torch.cat([rbf, sequence_adjacent.unsqueeze(-1).float(), confidence.unsqueeze(-1)], dim=-1)
        scores = scores + self.edge_mlp(edge).permute(0, 3, 1, 2).to(scores.dtype)
        scores = scores.masked_fill(~adjacency[:, None], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1) * adjacency[:, None].to(scores.dtype)
        attention = self.dropout(attention)
        self.last_center_attention = attention[:, :, 0].detach()
        update = torch.einsum("bhnm,bmhd->bnhd", attention, value).reshape(batch, nodes, width)
        hidden = hidden + self.dropout(self.output(update))
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden * node_mask.unsqueeze(-1).to(hidden.dtype)


class SequenceBackboneGraphLateFusion(nn.Module):
    """G1: frozen SequenceBackbone and independent two-layer GAT, joined only at the center."""

    model_name = "G1_SequenceBackbone_Independent_Graph_Late_Fusion_v1"

    def __init__(self) -> None:
        super().__init__()
        self.parent = SequenceBackbone()
        self.geometry_adapter = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 192))
        self.node_input_norm = nn.LayerNorm(192)
        self.graph_layers = nn.ModuleList([DenseStructureGATLayer() for _ in range(2)])
        self.graph_final_norm = nn.LayerNorm(192)
        self.graph_pool_score = nn.Sequential(nn.Linear(192, 64), nn.Tanh(), nn.Linear(64, 1))
        self.graph_summary = nn.Sequential(nn.Linear(384, 192), nn.GELU(), nn.LayerNorm(192))
        self.fusion_gate = nn.Sequential(nn.Linear(3 * 192 + 1, 192), nn.GELU(), nn.Linear(192, 192))
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.constant_(self.fusion_gate[-1].bias, -2.0)
        self.fusion_gamma = nn.Parameter(torch.zeros(()))
        self.freeze_parent()

    def load_parent_state(self, state: dict[str, torch.Tensor]) -> None:
        self.parent.load_state_dict(state, strict=True); self.freeze_parent()

    def freeze_parent(self) -> None:
        for parameter in self.parent.parameters(): parameter.requires_grad_(False)
        self.parent.eval()

    def train(self, mode: bool = True) -> "SequenceBackboneGraphLateFusion":
        super().train(mode); self.parent.eval(); return self

    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor,
               esm_embeddings: torch.Tensor, esm_present_mask: torch.Tensor,
               node_esm: torch.Tensor, node_positions: torch.Tensor,
               node_coordinates: torch.Tensor, node_geometry: torch.Tensor,
               node_mask: torch.Tensor) -> dict[str, Any]:
        sequence = self.parent.encode(token_ids, attention_mask, esm_embeddings, esm_present_mask)
        h_sequence = sequence["center_token"]
        node_mask = node_mask.bool()
        frozen_node_language = self.parent.esm_projector(node_esm, node_mask)
        graph = self.node_input_norm(frozen_node_language + self.geometry_adapter(node_geometry.float()))
        graph = graph * node_mask.unsqueeze(-1).to(graph.dtype)
        for layer in self.graph_layers:
            graph = layer(graph, node_coordinates, node_positions, node_geometry, node_mask)
        graph = self.graph_final_norm(graph) * node_mask.unsqueeze(-1).to(graph.dtype)
        pool_logits = self.graph_pool_score(graph).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~node_mask, torch.finfo(pool_logits.dtype).min)
        pool_weights = torch.softmax(pool_logits, dim=-1) * node_mask.to(pool_logits.dtype)
        pooled = (pool_weights.unsqueeze(-1) * graph).sum(1)
        graph_present = node_mask[:, 0]
        h_graph = self.graph_summary(torch.cat([graph[:, 0], pooled], dim=-1))
        h_graph = h_graph * graph_present.unsqueeze(-1).to(h_graph.dtype)
        missing = (~graph_present).unsqueeze(-1).to(h_sequence.dtype)
        gate = torch.sigmoid(self.fusion_gate(torch.cat(
            [h_sequence, h_graph, h_sequence * h_graph, missing], dim=-1
        ))) * graph_present.unsqueeze(-1).to(h_sequence.dtype)
        h_final = h_sequence + self.fusion_gamma.to(h_sequence.dtype) * gate * h_graph
        return {"center_token": h_final, "sequence_center_token": h_sequence,
                "graph_center_token": graph[:, 0], "graph_summary": h_graph,
                "graph_nodes": graph, "graph_pool_weights": pool_weights,
                "fusion_gate": gate, "graph_present": graph_present,
                "layer_center_attentions": [layer.last_center_attention for layer in self.graph_layers]}

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor,
                esm_embeddings: torch.Tensor, esm_present_mask: torch.Tensor,
                node_esm: torch.Tensor, node_positions: torch.Tensor,
                node_coordinates: torch.Tensor, node_geometry: torch.Tensor,
                node_mask: torch.Tensor) -> torch.Tensor:
        representation = self.encode(token_ids, attention_mask, esm_embeddings, esm_present_mask,
                                     node_esm, node_positions, node_coordinates, node_geometry, node_mask)
        return self.parent.backbone.classifier(representation["center_token"])[:, 0]

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
