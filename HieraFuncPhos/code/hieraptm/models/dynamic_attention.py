from __future__ import annotations

import math

import torch


def masked_attention(query, key, value, allowed, bias=None, dropout=None):
    score = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])

    if bias is not None:
        score = score + bias.to(score.dtype)

    permitted = allowed[:, None]
    weight = torch.softmax(
        score.float().masked_fill(~permitted, -1e9), dim=-1
    )
    weight = weight * permitted.float()
    weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    if dropout is not None:
        weight = dropout(weight)

    return weight.to(value.dtype) @ value, weight
