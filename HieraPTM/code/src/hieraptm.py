"""Controlled G2-G5 improvements over the frozen-SequenceBackbone G1 graph model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from graph_layers import DenseStructureGATLayer
from sequence_backbone import SequenceBackbone


VARIANTS = ("G2", "G3", "G4", "G5")


class MultiScaleStructureGATLayer(nn.Module):
    """GAT with distinct learned biases for 0-6, 6-8 and 8-12 Å edges."""

    def __init__(self, d_model: int = 192, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads; self.head_dim = d_model // heads; self.scale = self.head_dim ** -0.5
        self.input_norm = nn.LayerNorm(d_model); self.qkv = nn.Linear(d_model, 3 * d_model)
        self.edge_mlp = nn.Sequential(nn.Linear(13, 24), nn.GELU(), nn.Linear(24, heads))
        self.output = nn.Linear(d_model, d_model); self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 384), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(384, d_model), nn.Dropout(dropout))
        self.register_buffer("rbf_centers", torch.linspace(2.0, 12.0, 8), persistent=True)
        self.last_center_attention: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor, coordinates: torch.Tensor, positions: torch.Tensor,
                geometry: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        batch, nodes, width = hidden.shape
        qkv = self.qkv(self.input_norm(hidden)).view(batch, nodes, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        scores = torch.einsum("bnhd,bmhd->bhnm", query, key) * self.scale
        distance = torch.cdist(coordinates.float(), coordinates.float()).clamp(0, 32)
        sequence_adjacent = (positions[:, :, None] - positions[:, None, :]).abs().eq(1)
        eye = torch.eye(nodes, dtype=torch.bool, device=hidden.device).view(1, nodes, nodes)
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        adjacency = valid & (distance.le(12.0) | sequence_adjacent | eye)
        rbf = torch.exp(-0.5 * ((distance.unsqueeze(-1) - self.rbf_centers) / 1.5).square())
        rings = torch.stack((distance.le(6.0), distance.gt(6.0) & distance.le(8.0),
                             distance.gt(8.0) & distance.le(12.0)), dim=-1).float()
        confidence = torch.sqrt(geometry[:, :, 1].float().clamp(0, 1).unsqueeze(2)
                                * geometry[:, :, 1].float().clamp(0, 1).unsqueeze(1))
        edge = torch.cat([rbf, rings, sequence_adjacent.unsqueeze(-1).float(),
                          confidence.unsqueeze(-1)], dim=-1)
        scores = scores + self.edge_mlp(edge).permute(0, 3, 1, 2).to(scores.dtype)
        scores = scores.masked_fill(~adjacency[:, None], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1) * adjacency[:, None].to(scores.dtype)
        attention = self.dropout(attention); self.last_center_attention = attention[:, :, 0].detach()
        update = torch.einsum("bhnm,bmhd->bnhd", attention, value).reshape(batch, nodes, width)
        hidden = hidden + self.dropout(self.output(update))
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden * node_mask.unsqueeze(-1).to(hidden.dtype)


class HieraPTM(nn.Module):
    """Four isolated directions sharing the exact same frozen SequenceBackbone parent and graph cache."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        variant = variant.upper()
        if variant not in VARIANTS: raise ValueError(f"Unknown graph variant: {variant}")
        self.variant = variant; self.model_name = "HieraPTM"
        self.use_multiscale = variant in ("G2", "G5")
        self.use_query_pool = variant in ("G3", "G5")
        self.use_cross = variant in ("G4", "G5")
        self.use_late = variant in ("G2", "G3", "G5")
        self.parent = SequenceBackbone()
        self.geometry_adapter = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 192))
        self.node_input_norm = nn.LayerNorm(192)
        layer = MultiScaleStructureGATLayer if self.use_multiscale else DenseStructureGATLayer
        self.graph_layers = nn.ModuleList([layer() for _ in range(2)])
        self.graph_final_norm = nn.LayerNorm(192)
        if self.use_query_pool:
            self.graph_pool_score = nn.Sequential(nn.Linear(4 * 192, 192), nn.GELU(), nn.Linear(192, 1))
        else:
            self.graph_pool_score = nn.Sequential(nn.Linear(192, 64), nn.Tanh(), nn.Linear(64, 1))
        self.graph_summary = nn.Sequential(nn.Linear(384, 192), nn.GELU(), nn.LayerNorm(192))
        self.fusion_gate = nn.Sequential(nn.Linear(3 * 192 + 1, 192), nn.GELU(), nn.Linear(192, 192))
        nn.init.zeros_(self.fusion_gate[-1].weight); nn.init.constant_(self.fusion_gate[-1].bias, -2.0)
        self.fusion_gamma = nn.Parameter(torch.zeros(()), requires_grad=self.use_late)
        if not self.use_late:
            for parameter in self.fusion_gate.parameters(): parameter.requires_grad_(False)
            for parameter in self.graph_pool_score.parameters(): parameter.requires_grad_(False)
            for parameter in self.graph_summary.parameters(): parameter.requires_grad_(False)
        if self.use_cross:
            self.cross_query_norm = nn.LayerNorm(192); self.cross_graph_norm = nn.LayerNorm(192)
            self.cross_attention = nn.MultiheadAttention(192, 4, dropout=0.1, batch_first=True)
            self.cross_output_norm = nn.LayerNorm(192)
            self.cross_gamma = nn.Parameter(torch.zeros(()))
        else:
            self.cross_gamma = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.freeze_parent()

    def load_parent_state(self, state: dict[str, torch.Tensor]) -> None:
        self.parent.load_state_dict(state, strict=True); self.freeze_parent()

    def freeze_parent(self) -> None:
        for parameter in self.parent.parameters(): parameter.requires_grad_(False)
        self.parent.eval()

    def train(self, mode: bool = True) -> "HieraPTM":
        super().train(mode); self.parent.eval(); return self

    def _pool(self, graph: torch.Tensor, h_sequence: torch.Tensor,
              node_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_query_pool:
            query = h_sequence.unsqueeze(1).expand_as(graph)
            pool_input = torch.cat([graph, query, graph * query, (graph - query).abs()], dim=-1)
        else:
            pool_input = graph
        logits = self.graph_pool_score(pool_input).squeeze(-1)
        logits = logits.masked_fill(~node_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1) * node_mask.to(logits.dtype)
        return (weights.unsqueeze(-1) * graph).sum(1), weights

    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor,
               esm_embeddings: torch.Tensor, esm_present_mask: torch.Tensor,
               node_esm: torch.Tensor, node_positions: torch.Tensor,
               node_coordinates: torch.Tensor, node_geometry: torch.Tensor,
               node_mask: torch.Tensor) -> dict[str, Any]:
        sequence = self.parent.encode(token_ids, attention_mask, esm_embeddings, esm_present_mask)
        sequence_tokens = sequence["transformer_tokens"]; h_sequence = sequence["center_token"]
        node_mask = node_mask.bool(); graph_present = node_mask[:, 0]
        language = self.parent.esm_projector(node_esm, node_mask)
        graph = self.node_input_norm(language + self.geometry_adapter(node_geometry.float()))
        graph = graph * node_mask.unsqueeze(-1).to(graph.dtype)
        for layer in self.graph_layers:
            graph = layer(graph, node_coordinates, node_positions, node_geometry, node_mask)
        graph = self.graph_final_norm(graph) * node_mask.unsqueeze(-1).to(graph.dtype)
        pooled, pool_weights = self._pool(graph, h_sequence, node_mask)
        h_graph = self.graph_summary(torch.cat([graph[:, 0], pooled], dim=-1))
        h_graph = h_graph * graph_present.unsqueeze(-1).to(h_graph.dtype)
        missing = (~graph_present).unsqueeze(-1).to(h_sequence.dtype)
        gate = torch.sigmoid(self.fusion_gate(torch.cat(
            [h_sequence, h_graph, h_sequence * h_graph, missing], dim=-1
        ))) * graph_present.unsqueeze(-1).to(h_sequence.dtype)
        h_final = h_sequence
        if self.use_late:
            h_final = h_final + self.fusion_gamma.to(h_final.dtype) * gate * h_graph
        if self.use_cross:
            safe_mask = node_mask.clone(); safe_mask[~graph_present, 0] = True
            cross, cross_weights = self.cross_attention(
                self.cross_query_norm(sequence_tokens), self.cross_graph_norm(graph),
                self.cross_graph_norm(graph), key_padding_mask=~safe_mask,
                need_weights=True, average_attn_weights=False,
            )
            cross = self.cross_output_norm(cross) * graph_present[:, None, None].to(cross.dtype)
            h_final = h_final + self.cross_gamma.to(h_final.dtype) * cross[:, 30]
        else:
            cross_weights = None
        return {"center_token": h_final, "sequence_center_token": h_sequence,
                "graph_center_token": graph[:, 0], "graph_summary": h_graph,
                "graph_nodes": graph, "graph_pool_weights": pool_weights,
                "fusion_gate": gate, "graph_present": graph_present,
                "layer_center_attentions": [layer.last_center_attention for layer in self.graph_layers],
                "cross_attention_weights": cross_weights}

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


def build_graph_variant(variant: str) -> HieraPTM:
    return HieraPTM(variant)
