from __future__ import annotations

from typing import Any
import torch
from torch import nn
from graph_layers import DenseStructureGATLayer
from sequence_backbone import SequenceBackbone


class HieraPTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parent = SequenceBackbone()
        self.geometry_adapter = nn.Sequential(
            nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 192))
        self.node_input_norm = nn.LayerNorm(192)
        self.graph_layers = nn.ModuleList([
            DenseStructureGATLayer() for _ in range(2)])
        self.graph_final_norm = nn.LayerNorm(192)
        self.cross_query_norm = nn.LayerNorm(192)
        self.cross_graph_norm = nn.LayerNorm(192)
        self.cross_attention = nn.MultiheadAttention(
            192, 4, dropout=0.1, batch_first=True)
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

    def train(self, mode: bool = True) -> 'HieraPTM':
        super().train(mode)
        self.parent.eval()
        return self

    def encode(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor, esm_present_mask: torch.Tensor,
        node_esm: torch.Tensor, node_positions: torch.Tensor,
        node_coordinates: torch.Tensor, node_geometry: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, Any]:
        sequence = self.parent.encode(
            token_ids, attention_mask, esm_embeddings, esm_present_mask)
        sequence_tokens = sequence['transformer_tokens']
        sequence_center = sequence['center_token']
        node_mask = node_mask.bool()
        graph_present = node_mask[:, 0]
        language = self.parent.esm_projector(node_esm, node_mask)
        graph = self.node_input_norm(
            language + self.geometry_adapter(node_geometry.float()))
        graph = graph * node_mask.unsqueeze(-1).to(graph.dtype)
        for layer in self.graph_layers:
            graph = layer(
                graph, node_coordinates, node_positions, node_geometry, node_mask)
        graph = self.graph_final_norm(graph)
        graph = graph * node_mask.unsqueeze(-1).to(graph.dtype)
        safe_mask = node_mask.clone()
        safe_mask[~graph_present, 0] = True
        cross, attention = self.cross_attention(
            self.cross_query_norm(sequence_tokens),
            self.cross_graph_norm(graph),
            self.cross_graph_norm(graph),
            key_padding_mask=~safe_mask, need_weights=True,
            average_attn_weights=False)
        cross = self.cross_output_norm(cross)
        cross = cross * graph_present[:, None, None].to(cross.dtype)
        center = sequence_center + self.cross_gamma.to(sequence_center.dtype) * cross[:, 30]
        return {
            'center_token': center,
            'sequence_center_token': sequence_center,
            'graph_center_token': graph[:, 0],
            'graph_nodes': graph,
            'graph_present': graph_present,
            'layer_center_attentions': [
                layer.last_center_attention for layer in self.graph_layers],
            'cross_attention_weights': attention,
        }

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor, esm_present_mask: torch.Tensor,
        node_esm: torch.Tensor, node_positions: torch.Tensor,
        node_coordinates: torch.Tensor, node_geometry: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encode(
            token_ids, attention_mask, esm_embeddings, esm_present_mask,
            node_esm, node_positions, node_coordinates, node_geometry, node_mask)
        return self.parent.backbone.classifier(representation['center_token'])[:, 0]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
