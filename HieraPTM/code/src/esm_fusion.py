from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import torch
from torch import nn

@dataclass(frozen=True)
class ESMFusionSettings:
    vocabulary_size: int = 27
    padding_id: int = 0
    window_length: int = 61
    center_index: int = 30
    esm_dimension: int = 640
    esm_hidden_dimension: int = 256
    d_model: int = 192
    local_pool_radius: int = 8
    dropout: float = 0.1
    modality_dropout: float = 0.1
    gate_bias_initialization: float = -2.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'ESMFusionConfig':
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.vocabulary_size != 27 or self.padding_id != 0:
            raise ValueError('model must use the frozen input vocabulary')
        if self.window_length != 61 or self.center_index != 30:
            raise ValueError('model requires a centered 61-aa window')
        if (self.esm_dimension, self.esm_hidden_dimension, self.d_model) != (640, 256, 192):
            raise ValueError('ESM projection is frozen at 640 -> 256 -> 192')
        if self.local_pool_radius != 8:
            raise ValueError('ESM local pooling radius is frozen at 8')
        if self.dropout != 0.1 or self.modality_dropout != 0.1:
            raise ValueError('Dropout and modality dropout are frozen at 0.1')
        if self.gate_bias_initialization != -2.0:
            raise ValueError('ESM gate bias must be initialized to -2')

class ESMInputMixin:
    config: ESMFusionSettings

    def _validate_inputs(self, token_ids: torch.Tensor, attention_mask: torch.Tensor, esm_embeddings: torch.Tensor, esm_present_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = token_ids.shape[0] if token_ids.ndim else -1
        expected_tokens = (batch, self.config.window_length)
        expected_esm = (batch, self.config.window_length, self.config.esm_dimension)
        if token_ids.ndim != 2 or tuple(token_ids.shape) != expected_tokens:
            raise ValueError('token_ids must have shape [batch,61]')
        if token_ids.dtype != torch.long:
            raise TypeError('token_ids must be torch.int64')
        if tuple(attention_mask.shape) != expected_tokens:
            raise ValueError('attention_mask must have shape [batch,61]')
        if tuple(esm_embeddings.shape) != expected_esm:
            raise ValueError('esm_embeddings must have shape [batch,61,640]')
        if tuple(esm_present_mask.shape) != expected_tokens:
            raise ValueError('esm_present_mask must have shape [batch,61]')
        mask = attention_mask.bool()
        present = esm_present_mask.bool()
        if not bool(mask[:, self.config.center_index].all()):
            raise ValueError('center residue must be present in every sequence window')
        if bool((present & ~mask).any()):
            raise ValueError('esm_present_mask must be a subset of attention_mask')
        if present.any() and (not bool(torch.isfinite(esm_embeddings[present]).all())):
            raise ValueError('present ESM values must all be finite')
        return (mask, present)

class ESMTokenProjector(nn.Module):

    def __init__(self, config: ESMFusionSettings) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.esm_dimension, config.esm_hidden_dimension)
        self.activation = nn.GELU()
        self.output = nn.Linear(config.esm_hidden_dimension, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, esm_embeddings: torch.Tensor, present_mask: torch.Tensor) -> torch.Tensor:
        mask = present_mask.bool().unsqueeze(-1)
        clean = torch.where(mask, esm_embeddings.float(), torch.zeros_like(esm_embeddings.float()))
        projected = self.norm(self.output(self.activation(self.input(clean))))
        return projected * mask.to(projected.dtype)
