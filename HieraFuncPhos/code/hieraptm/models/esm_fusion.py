"""Frozen E0/E1/E2 ESM-fusion model interfaces for Revised Day 33.

This module deliberately contains no ESM-2 network.  It consumes the frozen
per-residue ESM cache validated on Revised Days 21--24.  The selected C1
backbone is imported unchanged so its earlier provenance remains intact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .backbones import C1CNNRelativeTransformer, C1Config


@dataclass(frozen=True)
class ESMFusionConfig:
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
    def from_dict(cls, value: dict[str, Any]) -> "ESMFusionConfig":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.vocabulary_size != 27 or self.padding_id != 0:
            raise ValueError("Day33 must use the frozen Day2 vocabulary")
        if self.window_length != 61 or self.center_index != 30:
            raise ValueError("Day33 requires a centered 61-aa window")
        if (self.esm_dimension, self.esm_hidden_dimension, self.d_model) != (640, 256, 192):
            raise ValueError("ESM projection is frozen at 640 -> 256 -> 192")
        if self.local_pool_radius != 8:
            raise ValueError("ESM local pooling radius is frozen at 8")
        if self.dropout != 0.1 or self.modality_dropout != 0.1:
            raise ValueError("Dropout and modality dropout are frozen at 0.1")
        if self.gate_bias_initialization != -2.0:
            raise ValueError("E2 gate bias must be initialized to -2")


def _c1_config(config: ESMFusionConfig) -> C1Config:
    return C1Config(
        vocabulary_size=config.vocabulary_size,
        padding_id=config.padding_id,
        window_length=config.window_length,
        center_index=config.center_index,
        d_model=config.d_model,
    )


class ESMInputMixin:
    config: ESMFusionConfig

    def _validate_inputs(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor,
        esm_present_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = token_ids.shape[0] if token_ids.ndim else -1
        expected_tokens = (batch, self.config.window_length)
        expected_esm = (batch, self.config.window_length, self.config.esm_dimension)
        if token_ids.ndim != 2 or tuple(token_ids.shape) != expected_tokens:
            raise ValueError("token_ids must have shape [batch,61]")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must be torch.int64")
        if tuple(attention_mask.shape) != expected_tokens:
            raise ValueError("attention_mask must have shape [batch,61]")
        if tuple(esm_embeddings.shape) != expected_esm:
            raise ValueError("esm_embeddings must have shape [batch,61,640]")
        if tuple(esm_present_mask.shape) != expected_tokens:
            raise ValueError("esm_present_mask must have shape [batch,61]")
        mask = attention_mask.bool()
        present = esm_present_mask.bool()
        if not bool(mask[:, self.config.center_index].all()):
            raise ValueError("center residue must be present in every sequence window")
        if bool((present & ~mask).any()):
            raise ValueError("esm_present_mask must be a subset of attention_mask")
        if present.any() and not bool(torch.isfinite(esm_embeddings[present]).all()):
            raise ValueError("present ESM values must all be finite")
        return mask, present


class ESMTokenProjector(nn.Module):
    """Trainable projection for frozen float16 cache values."""

    def __init__(self, config: ESMFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.esm_dimension, config.esm_hidden_dimension)
        self.activation = nn.GELU()
        self.output = nn.Linear(config.esm_hidden_dimension, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, esm_embeddings: torch.Tensor, present_mask: torch.Tensor) -> torch.Tensor:
        mask = present_mask.bool().unsqueeze(-1)
        # torch.where is intentional: NaN * 0 is still NaN for an absent cache row.
        clean = torch.where(mask, esm_embeddings.float(), torch.zeros_like(esm_embeddings.float()))
        projected = self.norm(self.output(self.activation(self.input(clean))))
        return projected * mask.to(projected.dtype)


def _pool_center_and_local(
    tokens: torch.Tensor,
    present_mask: torch.Tensor,
    center_index: int,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = center_index - radius
    stop = center_index + radius + 1
    center = tokens[:, center_index]
    local_tokens = tokens[:, start:stop]
    local_mask = present_mask[:, start:stop].unsqueeze(-1).to(tokens.dtype)
    denominator = local_mask.sum(dim=1).clamp_min(1.0)
    local_mean = (local_tokens * local_mask).sum(dim=1) / denominator
    pooled = torch.cat([center, local_mean], dim=-1)
    return pooled, center, local_mean


class E0ESMPooledProbe(ESMInputMixin, nn.Module):
    """E0: frozen-ESM center/local pooling plus a lightweight MLP probe."""

    model_name = "E0_ESM_Center_Local_MLP_v1"

    def __init__(self, config: ESMFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or ESMFusionConfig()
        self.config.validate()
        self.esm_projector = ESMTokenProjector(self.config)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * self.config.d_model),
            nn.Linear(2 * self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, 1),
        )

    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor,
        esm_present_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask, present = self._validate_inputs(
            token_ids, attention_mask, esm_embeddings, esm_present_mask
        )
        del mask
        projected = self.esm_projector(esm_embeddings, present)
        pooled, center, local = _pool_center_and_local(
            projected, present, self.config.center_index, self.config.local_pool_radius
        )
        return {
            "projected_esm_tokens": projected,
            "esm_center_token": center,
            "esm_local_mean": local,
            "pooled_esm": pooled,
        }

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor,
        esm_present_mask: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encode(
            token_ids, attention_mask, esm_embeddings, esm_present_mask
        )
        return self.classifier(representation["pooled_esm"])[:, 0]


class E1C1LateESMFusion(ESMInputMixin, nn.Module):
    """E1: C1 center token and independently pooled ESM are fused at the head."""

    model_name = "E1_C1_Late_ESM_Fusion_v1"

    def __init__(self, config: ESMFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or ESMFusionConfig()
        self.config.validate()
        self.backbone = C1CNNRelativeTransformer(_c1_config(self.config))
        # The original one-logit head is not part of E1; keeping it would add
        # dead trainable parameters that never receive gradients.
        self.backbone.classifier = nn.Identity()
        self.esm_projector = ESMTokenProjector(self.config)
        late_dimension = 3 * self.config.d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(late_dimension),
            nn.Linear(late_dimension, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, 1),
        )

    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor,
        esm_present_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        mask, present = self._validate_inputs(
            token_ids, attention_mask, esm_embeddings, esm_present_mask
        )
        backbone = self.backbone.encode(token_ids, mask, return_attention=return_attention)
        projected = self.esm_projector(esm_embeddings, present)
        pooled, center, local = _pool_center_and_local(
            projected, present, self.config.center_index, self.config.local_pool_radius
        )
        late = torch.cat([backbone["center_token"], pooled], dim=-1)
        return {
            **backbone,
            "projected_esm_tokens": projected,
            "esm_center_token": center,
            "esm_local_mean": local,
            "late_fusion_token": late,
        }

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor,
        esm_present_mask: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encode(
            token_ids, attention_mask, esm_embeddings, esm_present_mask
        )
        return self.classifier(representation["late_fusion_token"])[:, 0]


class E2C1TokenGatedESMFusion(ESMInputMixin, nn.Module):
    """E2: token-wise channel gate inserts ESM before the C1 Transformer."""

    model_name = "E2_C1_Token_Gated_ESM_Fusion_v1"

    def __init__(self, config: ESMFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or ESMFusionConfig()
        self.config.validate()
        self.backbone = C1CNNRelativeTransformer(_c1_config(self.config))
        self.esm_projector = ESMTokenProjector(self.config)
        self.gate_relative_position_embedding = nn.Embedding(
            self.config.window_length, self.config.d_model
        )
        gate_input_dimension = 4 * self.config.d_model + 1
        self.gate = nn.Linear(gate_input_dimension, self.config.d_model)
        nn.init.constant_(self.gate.bias, self.config.gate_bias_initialization)
        relative_ids = torch.arange(self.config.window_length, dtype=torch.long)
        self.register_buffer("gate_relative_position_ids", relative_ids, persistent=False)

    def _apply_modality_dropout(
        self, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = present.shape[0]
        dropped = torch.zeros(batch, dtype=torch.bool, device=present.device)
        if self.training and self.config.modality_dropout > 0:
            dropped = torch.rand(batch, device=present.device).lt(self.config.modality_dropout)
        effective = present & ~dropped.unsqueeze(1)
        return effective, dropped

    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor,
        esm_present_mask: torch.Tensor,
        return_attention: bool = False,
        force_gate_zero: bool = False,
    ) -> dict[str, Any]:
        mask, present = self._validate_inputs(
            token_ids, attention_mask, esm_embeddings, esm_present_mask
        )
        effective_present, modality_dropped = self._apply_modality_dropout(present)
        motif = self.backbone.token_cnn(token_ids, mask)
        projected = self.esm_projector(esm_embeddings, effective_present)
        center_motif = motif[:, self.config.center_index].unsqueeze(1).expand(-1, 61, -1)
        relative = self.gate_relative_position_embedding(
            self.gate_relative_position_ids.unsqueeze(0).expand(token_ids.shape[0], -1)
        )
        missing = (~effective_present).unsqueeze(-1).to(motif.dtype)
        gate_input = torch.cat([motif, projected, center_motif, relative, missing], dim=-1)
        raw_gate = torch.sigmoid(self.gate(gate_input))
        effective_gate = raw_gate * effective_present.unsqueeze(-1).to(raw_gate.dtype)
        if force_gate_zero:
            effective_gate = torch.zeros_like(effective_gate)
        fused = motif + effective_gate * projected
        fused = fused * mask.unsqueeze(-1).to(fused.dtype)
        hidden, attentions = self.backbone._run_transformer(
            fused, mask, return_attention=return_attention
        )
        gate_position_mean = effective_gate.mean(dim=-1)
        local_start = self.config.center_index - self.config.local_pool_radius
        local_stop = self.config.center_index + self.config.local_pool_radius + 1
        local_mask = effective_present[:, local_start:local_stop].to(gate_position_mean.dtype)
        local_denominator = local_mask.sum(dim=1).clamp_min(1.0)
        local_gate_mean = (
            gate_position_mean[:, local_start:local_stop] * local_mask
        ).sum(dim=1) / local_denominator
        return {
            "motif_tokens": motif,
            "projected_esm_tokens": projected,
            "raw_esm_gate_tokens": raw_gate,
            "esm_gate_tokens": effective_gate,
            "esm_gate_position_mean": gate_position_mean,
            "esm_gate_center_mean": gate_position_mean[:, self.config.center_index],
            "esm_gate_local_mean": local_gate_mean,
            "esm_present_mask_after_dropout": effective_present,
            "modality_dropped": modality_dropped,
            "fused_tokens": fused,
            "transformer_tokens": hidden,
            "center_token": hidden[:, self.config.center_index],
            "attentions": attentions,
        }

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        esm_embeddings: torch.Tensor,
        esm_present_mask: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encode(
            token_ids, attention_mask, esm_embeddings, esm_present_mask
        )
        return self.backbone.classifier(representation["center_token"])[:, 0]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
