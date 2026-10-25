"""Additive modular DeepMVP models introduced after Revised Day37."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from esm_fusion import (
    ESMFusionConfig,
    ESMInputMixin,
    ESMTokenProjector,
)
from backbones import (
    C2CNNBiGRURelativeTransformer,
    C2Config,
)


@dataclass(frozen=True)
class SequenceBackboneConfig(ESMFusionConfig):
    bigru_hidden_per_direction: int = 64

    def validate(self) -> None:
        super().validate()
        if self.bigru_hidden_per_direction != 64:
            raise ValueError("SequenceBackbone BiGRU hidden size is frozen at 64 per direction")


def _c2_config(config: SequenceBackboneConfig) -> C2Config:
    return C2Config(
        vocabulary_size=config.vocabulary_size,
        padding_id=config.padding_id,
        window_length=config.window_length,
        center_index=config.center_index,
        d_model=config.d_model,
        bigru_hidden_per_direction=config.bigru_hidden_per_direction,
    )


class SequenceBackbone(ESMInputMixin, nn.Module):
    """CNN -> BiGRU -> E2 gated frozen ESM -> center-aware Transformer."""

    model_name = "Sequence Backbone"

    def __init__(self, config: SequenceBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or SequenceBackboneConfig()
        self.config.validate()
        self.backbone = C2CNNBiGRURelativeTransformer(_c2_config(self.config))
        self.esm_projector = ESMTokenProjector(self.config)
        self.gate_relative_position_embedding = nn.Embedding(
            self.config.window_length, self.config.d_model
        )
        self.gate = nn.Linear(4 * self.config.d_model + 1, self.config.d_model)
        nn.init.constant_(self.gate.bias, self.config.gate_bias_initialization)
        self.register_buffer(
            "gate_relative_position_ids",
            torch.arange(self.config.window_length, dtype=torch.long),
            persistent=False,
        )

    def _apply_modality_dropout(
        self, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dropped = torch.zeros(
            present.shape[0], dtype=torch.bool, device=present.device
        )
        if self.training and self.config.modality_dropout > 0:
            dropped = torch.rand(present.shape[0], device=present.device).lt(
                self.config.modality_dropout
            )
        return present & ~dropped.unsqueeze(1), dropped

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
        bigru_context, contextual = self.backbone._contextualize_motifs(motif, mask)
        projected_esm = self.esm_projector(esm_embeddings, effective_present)
        center_context = contextual[:, self.config.center_index].unsqueeze(1).expand(
            -1, self.config.window_length, -1
        )
        relative = self.gate_relative_position_embedding(
            self.gate_relative_position_ids.unsqueeze(0).expand(token_ids.shape[0], -1)
        )
        missing = (~effective_present).unsqueeze(-1).to(contextual.dtype)
        gate_input = torch.cat(
            [contextual, projected_esm, center_context, relative, missing], dim=-1
        )
        raw_gate = torch.sigmoid(self.gate(gate_input))
        effective_gate = raw_gate * effective_present.unsqueeze(-1).to(raw_gate.dtype)
        if force_gate_zero:
            effective_gate = torch.zeros_like(effective_gate)
        fused = contextual + effective_gate * projected_esm
        fused = fused * mask.unsqueeze(-1).to(fused.dtype)
        hidden, attentions = self.backbone._run_transformer(
            fused, mask, return_attention=return_attention
        )
        gate_position_mean = effective_gate.mean(dim=-1)
        local_start = self.config.center_index - self.config.local_pool_radius
        local_stop = self.config.center_index + self.config.local_pool_radius + 1
        local_mask = effective_present[:, local_start:local_stop].to(
            gate_position_mean.dtype
        )
        local_denominator = local_mask.sum(dim=1).clamp_min(1.0)
        local_gate_mean = (
            gate_position_mean[:, local_start:local_stop] * local_mask
        ).sum(dim=1) / local_denominator
        return {
            "motif_tokens": motif,
            "bigru_context_tokens": bigru_context,
            "bigru_projected_tokens": contextual,
            "projected_esm_tokens": projected_esm,
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
