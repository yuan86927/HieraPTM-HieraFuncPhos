"""Numerical components extracted from the frozen final Task2 snapshot."""
from __future__ import annotations
import torch
from torch import nn

class QVLoRALinear(nn.Module):
    """Preserve a packed QKV projection and add updates to Q and V only."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.05) -> None:
        super().__init__()
        if (base.in_features, base.out_features) != (192, 576):
            raise ValueError("Expected Task1 packed qkv projection 192 -> 576")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.drop = nn.Dropout(dropout)
        self.q_a = nn.Linear(192, rank, bias=False)
        self.q_b = nn.Linear(rank, 192, bias=False)
        self.v_a = nn.Linear(192, rank, bias=False)
        self.v_b = nn.Linear(rank, 192, bias=False)
        for matrix in (self.q_a.weight, self.v_a.weight):
            nn.init.kaiming_uniform_(matrix, a=5 ** 0.5)
        nn.init.zeros_(self.q_b.weight)
        nn.init.zeros_(self.v_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden)
        dropped = self.drop(hidden)
        delta_q = self.q_b(self.q_a(dropped)) * self.scale
        delta_v = self.v_b(self.v_a(dropped)) * self.scale
        return base + torch.cat([delta_q, torch.zeros_like(delta_q), delta_v], dim=-1)

class ZeroInitLoRALinear(nn.Module):
    """Frozen base linear plus a zero-initialized low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = 2.0  # alpha = 2 * rank
        self.drop = nn.Dropout(dropout)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.base(hidden) + self.b(self.a(self.drop(hidden))) * self.scale
