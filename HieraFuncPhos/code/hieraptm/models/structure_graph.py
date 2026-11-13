
"""Structural graph attention for the HieraPTM encoder."""

from __future__ import annotations

import torch
from torch import nn


class DenseStructureGATLayer(nn.Module):
    """Update residue representations using geometry-aware graph attention."""

    def __init__(
        self,
        d_model: int = 192,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim ** -0.5

        self.input_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.edge_mlp = nn.Sequential(
            nn.Linear(10, 16),
            nn.GELU(),
            nn.Linear(16, heads),
        )
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, d_model),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "rbf_centers", torch.linspace(2.0, 12.0, 8), persistent=True
        )
        self.last_center_attention: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        coordinates: torch.Tensor,
        positions: torch.Tensor,
        geometry: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes, width = hidden.shape
        normalized = self.input_norm(hidden)
        qkv = self.qkv(normalized).view(
            batch, nodes, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(2)
        scores = torch.einsum("bnhd,bmhd->bhnm", query, key) * self.scale

        distance = torch.cdist(
            coordinates.float(), coordinates.float()
        ).clamp(0, 32)
        sequence_adjacent = (
            positions[:, :, None] - positions[:, None, :]
        ).abs().eq(1)
        eye = torch.eye(
            nodes, dtype=torch.bool, device=hidden.device
        ).view(1, nodes, nodes)
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        adjacency = valid & (distance.le(8.0) | sequence_adjacent | eye)

        rbf = torch.exp(
            -0.5 * (
                (distance.unsqueeze(-1) - self.rbf_centers) / 1.5
            ).square()
        )
        confidence = torch.sqrt(
            geometry[:, :, 1].float().clamp(0, 1).unsqueeze(2)
            * geometry[:, :, 1].float().clamp(0, 1).unsqueeze(1)
        )
        edge = torch.cat(
            [
                rbf,
                sequence_adjacent.unsqueeze(-1).float(),
                confidence.unsqueeze(-1),
            ],
            dim=-1,
        )
        scores = scores + self.edge_mlp(edge).permute(0, 3, 1, 2).to(scores.dtype)
        scores = scores.masked_fill(
            ~adjacency[:, None], torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=-1) * adjacency[:, None].to(
            scores.dtype
        )
        attention = self.dropout(attention)
        self.last_center_attention = attention[:, :, 0].detach()

        update = torch.einsum(
            "bhnm,bmhd->bnhd", attention, value
        ).reshape(batch, nodes, width)
        hidden = hidden + self.dropout(self.output(update))
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden * node_mask.unsqueeze(-1).to(hidden.dtype)
