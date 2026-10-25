"""Sequence backbones for the revised DeepMVP-ESM-MoEFormer study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class D0Config:
    vocabulary_size: int = 27
    padding_id: int = 0
    window_length: int = 61
    center_index: int = 30
    embedding_dimension: int = 64
    cnn_kernels: tuple[int, ...] = (3, 5, 7)
    cnn_channels_per_kernel: int = 64
    cnn_dropout: float = 0.1
    gru_hidden_per_direction: int = 64
    classifier_hidden_dimension: int = 192
    classifier_dropout: float = 0.1

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "D0Config":
        local = dict(value)
        if "cnn_kernels" in local:
            local["cnn_kernels"] = tuple(local["cnn_kernels"])
        return cls(**local)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cnn_kernels"] = list(self.cnn_kernels)
        return value

    @property
    def motif_dimension(self) -> int:
        return len(self.cnn_kernels) * self.cnn_channels_per_kernel

    @property
    def gru_dimension(self) -> int:
        return 2 * self.gru_hidden_per_direction

    def validate(self) -> None:
        if self.vocabulary_size != 27 or self.padding_id != 0:
            raise ValueError("D0 must use the frozen Day2 vocabulary contract")
        if self.window_length != 61 or self.center_index != 30:
            raise ValueError("D0 requires a centered 61-aa window")
        if self.embedding_dimension != 64:
            raise ValueError("D0 AA embedding dimension is frozen at 64")
        if self.cnn_kernels != (3, 5, 7) or self.cnn_channels_per_kernel != 64:
            raise ValueError("D0 CNN is frozen at kernels 3/5/7 with 64 channels each")
        if any(kernel % 2 == 0 for kernel in self.cnn_kernels):
            raise ValueError("All same-padding CNN kernels must be odd")
        if self.gru_hidden_per_direction != 64:
            raise ValueError("D0 BiGRU is frozen at 64 hidden units per direction")


class DeepMVPStyleTokenCNN(nn.Module):
    """Multi-scale same-padding CNN that preserves one motif vector per residue."""

    def __init__(self, config: D0Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocabulary_size,
            config.embedding_dimension,
            padding_idx=config.padding_id,
        )
        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                config.embedding_dimension,
                config.cnn_channels_per_kernel,
                kernel_size=kernel,
                padding=kernel // 2,
            )
            for kernel in config.cnn_kernels
        ])
        self.norm = nn.LayerNorm(config.motif_dimension)
        self.dropout = nn.Dropout(config.cnn_dropout)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("token_ids and attention_mask must have shape [batch,61]")
        if token_ids.shape != attention_mask.shape or token_ids.shape[1] != self.config.window_length:
            raise ValueError("token_ids and attention_mask must share frozen [batch,61] shape")
        mask = attention_mask.bool()
        if not bool(mask[:, self.config.center_index].all()):
            raise ValueError("Every sample must have a valid center residue at index 30")
        if token_ids.numel() and (int(token_ids.min()) < 0 or int(token_ids.max()) >= self.config.vocabulary_size):
            raise ValueError("token_ids contain an ID outside the frozen vocabulary")
        embedded = self.embedding(token_ids) * mask.unsqueeze(-1).to(torch.float32)
        channel_first = embedded.transpose(1, 2)
        motif = torch.cat([F.gelu(layer(channel_first)) for layer in self.convolutions], dim=1).transpose(1, 2)
        motif = self.dropout(self.norm(motif))
        return motif * mask.unsqueeze(-1).to(motif.dtype)


class D0DeepMVPStyleCNNBiGRU(nn.Module):
    """Single-model, mask-safe DeepMVP-style CNN+BiGRU sequence baseline."""

    model_name = "D0_DeepMVP_style_CNN_BiGRU_v1"

    def __init__(self, config: D0Config | None = None) -> None:
        super().__init__()
        self.config = config or D0Config()
        self.config.validate()
        self.token_cnn = DeepMVPStyleTokenCNN(self.config)
        self.bigru = nn.GRU(
            input_size=self.config.motif_dimension,
            hidden_size=self.config.gru_hidden_per_direction,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        fusion_dimension = self.config.gru_dimension + 3 * self.config.motif_dimension
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dimension),
            nn.Linear(fusion_dimension, self.config.classifier_hidden_dimension),
            nn.GELU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.classifier_hidden_dimension, 1),
        )

    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = attention_mask.bool()
        motif = self.token_cnn(token_ids, mask)
        lengths = mask.sum(dim=1).to(torch.long)
        left_padding = torch.argmax(mask.to(torch.long), dim=1)
        compact = motif.new_zeros(motif.shape)
        for index in range(motif.shape[0]):
            length = int(lengths[index])
            start = int(left_padding[index])
            compact[index, :length] = motif[index, start : start + length]
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_context, _ = self.bigru(packed)
        compact_context, _ = nn.utils.rnn.pad_packed_sequence(
            packed_context, batch_first=True, total_length=self.config.window_length
        )
        context = motif.new_zeros(
            motif.shape[0], self.config.window_length, self.config.gru_dimension
        )
        for index in range(motif.shape[0]):
            length = int(lengths[index])
            start = int(left_padding[index])
            context[index, start : start + length] = compact_context[index, :length]
        context = context * mask.unsqueeze(-1).to(context.dtype)
        center_context = context[:, self.config.center_index]
        center_motif = motif[:, self.config.center_index]
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(motif.dtype)
        mean_motif = (motif * mask.unsqueeze(-1).to(motif.dtype)).sum(dim=1) / denominator
        maximum_motif = motif.masked_fill(~mask.unsqueeze(-1), torch.finfo(motif.dtype).min).amax(dim=1)
        return {
            "motif_tokens": motif,
            "context_tokens": context,
            "center_context": center_context,
            "center_motif": center_motif,
            "mean_motif": mean_motif,
            "maximum_motif": maximum_motif,
        }

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        representations = self.encode(token_ids, attention_mask)
        fused = torch.cat([
            representations["center_context"],
            representations["center_motif"],
            representations["mean_motif"],
            representations["maximum_motif"],
        ], dim=-1)
        return self.classifier(fused)[:, 0]


@dataclass(frozen=True)
class C1Config:
    vocabulary_size: int = 27
    padding_id: int = 0
    window_length: int = 61
    center_index: int = 30
    embedding_dimension: int = 64
    cnn_kernels: tuple[int, ...] = (3, 5, 7)
    cnn_channels_per_kernel: int = 64
    d_model: int = 192
    layers: int = 4
    attention_heads: int = 6
    d_ff: int = 512
    attention_pattern: tuple[str, ...] = ("local", "global", "local", "global")
    local_radius: int = 8
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "C1Config":
        local = dict(value)
        for key in ("cnn_kernels", "attention_pattern"):
            if key in local:
                local[key] = tuple(local[key])
        return cls(**local)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cnn_kernels"] = list(self.cnn_kernels)
        value["attention_pattern"] = list(self.attention_pattern)
        return value

    def validate(self) -> None:
        if self.vocabulary_size != 27 or self.padding_id != 0:
            raise ValueError("C1 must use the frozen Day2 vocabulary contract")
        if self.window_length != 61 or self.center_index != 30:
            raise ValueError("C1 requires a centered 61-aa window")
        if self.embedding_dimension != 64:
            raise ValueError("C1 AA embedding dimension is frozen at 64")
        if self.cnn_kernels != (3, 5, 7) or self.cnn_channels_per_kernel != 64:
            raise ValueError("C1 CNN is frozen at kernels 3/5/7 with 64 channels each")
        if self.d_model != 192 or self.layers != 4 or self.attention_heads != 6 or self.d_ff != 512:
            raise ValueError("C1 Transformer is frozen at d_model=192, 4 layers, 6 heads and d_ff=512")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.attention_pattern != ("local", "global", "local", "global"):
            raise ValueError("C1 attention pattern must be local/global/local/global")
        if self.local_radius != 8:
            raise ValueError("C1 local attention radius is frozen at 8")


class SignedRelativeCenterBias(nn.Module):
    """Per-head signed-distance buckets plus an explicit center-relation bias."""

    def __init__(self, heads: int, length: int, center_index: int) -> None:
        super().__init__()
        self.heads = heads
        self.length = length
        self.center_index = center_index
        self.relative_bias = nn.Embedding(2 * length - 1, heads)
        self.center_bias = nn.Embedding(4, heads)
        positions = torch.arange(length, dtype=torch.long)
        # score(i,j) uses signed key-minus-query distance j-i.
        buckets = positions.unsqueeze(0) - positions.unsqueeze(1) + length - 1
        query_center = positions.eq(center_index).view(length, 1)
        key_center = positions.eq(center_index).view(1, length)
        center_relation = query_center.to(torch.long) + 2 * key_center.to(torch.long)
        self.register_buffer("relative_bucket_ids", buckets, persistent=False)
        self.register_buffer("center_relation_ids", center_relation, persistent=False)

    def forward(self) -> torch.Tensor:
        relative = self.relative_bias(self.relative_bucket_ids)
        center = self.center_bias(self.center_relation_ids)
        return (relative + center).permute(2, 0, 1)


class RelativeCenterSelfAttention(nn.Module):
    """Mask-safe self-attention with PTM-centered structural bias."""

    def __init__(self, config: C1Config, pattern: str) -> None:
        super().__init__()
        if pattern not in {"local", "global"}:
            raise ValueError("attention pattern must be local or global")
        self.config = config
        self.pattern = pattern
        self.head_dimension = config.d_model // config.attention_heads
        self.scale = self.head_dimension**-0.5
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.bias = SignedRelativeCenterBias(
            config.attention_heads, config.window_length, config.center_index
        )
        positions = torch.arange(config.window_length)
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        allowed = torch.ones_like(distance, dtype=torch.bool)
        if pattern == "local":
            allowed = distance.le(config.local_radius)
            allowed[config.center_index, :] = True
        self.register_buffer("structural_allowed", allowed, persistent=False)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, width = hidden.shape
        if length != self.config.window_length or width != self.config.d_model:
            raise ValueError("C1 attention requires hidden shape [batch,61,192]")
        qkv = self.qkv(hidden).view(
            batch, length, 3, self.config.attention_heads, self.head_dimension
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores + self.bias().unsqueeze(0).to(scores.dtype)
        allowed = self.structural_allowed.view(1, 1, length, length)
        valid_keys = attention_mask.bool().view(batch, 1, 1, length)
        valid_queries = attention_mask.bool().view(batch, 1, length, 1)
        permitted = allowed & valid_keys & valid_queries
        scores = scores.masked_fill(~permitted, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        # A padded query can have no permitted key. With a finite masking value,
        # softmax would otherwise turn that row into a meaningless uniform
        # distribution. Explicit masking also makes the attention audit exact.
        probabilities = probabilities * permitted.to(probabilities.dtype)
        probabilities = self.attention_dropout(probabilities)
        context = torch.matmul(probabilities, value).transpose(1, 2).contiguous().view(
            batch, length, width
        )
        context = self.output_dropout(self.output(context))
        context = context * attention_mask.unsqueeze(-1).to(context.dtype)
        return context, probabilities if return_attention else None


class C1TransformerBlock(nn.Module):
    def __init__(self, config: C1Config, pattern: str) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = RelativeCenterSelfAttention(config, pattern)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        update, probabilities = self.attention(
            self.attention_norm(hidden), attention_mask, return_attention
        )
        hidden = hidden + update
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, probabilities


class C1CNNRelativeTransformer(nn.Module):
    """Token CNN followed by local/global center-aware relative Transformer."""

    model_name = "C1_CNN_Relative_Center_Transformer_v1"

    def __init__(self, config: C1Config | None = None) -> None:
        super().__init__()
        self.config = config or C1Config()
        self.config.validate()
        cnn_config = D0Config(
            vocabulary_size=self.config.vocabulary_size,
            padding_id=self.config.padding_id,
            window_length=self.config.window_length,
            center_index=self.config.center_index,
            embedding_dimension=self.config.embedding_dimension,
            cnn_kernels=self.config.cnn_kernels,
            cnn_channels_per_kernel=self.config.cnn_channels_per_kernel,
            cnn_dropout=self.config.dropout,
        )
        self.token_cnn = DeepMVPStyleTokenCNN(cnn_config)
        self.relative_position_embedding = nn.Embedding(self.config.window_length, self.config.d_model)
        self.center_marker_embedding = nn.Embedding(2, self.config.d_model)
        positions = torch.arange(self.config.window_length, dtype=torch.long)
        markers = torch.zeros(self.config.window_length, dtype=torch.long)
        markers[self.config.center_index] = 1
        self.register_buffer("position_ids", positions, persistent=False)
        self.register_buffer("center_marker_ids", markers, persistent=False)
        self.input_dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList([
            C1TransformerBlock(self.config, pattern)
            for pattern in self.config.attention_pattern
        ])
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.classifier = nn.Linear(self.config.d_model, 1)

    def _run_transformer(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        mask = attention_mask.bool()
        positions = self.position_ids.unsqueeze(0).expand(input_tokens.shape[0], -1)
        markers = self.center_marker_ids.unsqueeze(0).expand(input_tokens.shape[0], -1)
        hidden = input_tokens + self.relative_position_embedding(positions) + self.center_marker_embedding(markers)
        hidden = self.input_dropout(hidden) * mask.unsqueeze(-1).to(hidden.dtype)
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, probabilities = block(hidden, mask, return_attention)
            if probabilities is not None:
                attentions.append(probabilities)
        hidden = self.final_norm(hidden) * mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, attentions

    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        mask = attention_mask.bool()
        motif = self.token_cnn(token_ids, mask)
        hidden, attentions = self._run_transformer(motif, mask, return_attention)
        return {
            "motif_tokens": motif,
            "transformer_tokens": hidden,
            "center_token": hidden[:, self.config.center_index],
            "attentions": attentions,
        }

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        representations = self.encode(token_ids, attention_mask, return_attention=False)
        return self.classifier(representations["center_token"])[:, 0]


@dataclass(frozen=True)
class C2Config(C1Config):
    bigru_hidden_per_direction: int = 64

    def validate(self) -> None:
        super().validate()
        if self.bigru_hidden_per_direction != 64:
            raise ValueError("C2 BiGRU is frozen at 64 hidden units per direction")

    @property
    def bigru_dimension(self) -> int:
        return 2 * self.bigru_hidden_per_direction


class C2CNNBiGRURelativeTransformer(C1CNNRelativeTransformer):
    """CNN motifs contextualized by BiGRU before the frozen C1 Transformer."""

    model_name = "C2_CNN_BiGRU_Relative_Center_Transformer_v1"

    def __init__(self, config: C2Config | None = None) -> None:
        spec = config or C2Config()
        spec.validate()
        super().__init__(spec)
        self.config = spec
        self.bigru = nn.GRU(
            input_size=spec.d_model,
            hidden_size=spec.bigru_hidden_per_direction,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bigru_projection = nn.Linear(spec.bigru_dimension, spec.d_model)
        self.bigru_residual_norm = nn.LayerNorm(spec.d_model)
        self.bigru_dropout = nn.Dropout(spec.dropout)

    def _contextualize_motifs(
        self, motif: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = attention_mask.bool()
        lengths = mask.sum(dim=1).to(torch.long)
        left_padding = torch.argmax(mask.to(torch.long), dim=1)
        compact = motif.new_zeros(motif.shape)
        for index in range(motif.shape[0]):
            length = int(lengths[index])
            start = int(left_padding[index])
            compact[index, :length] = motif[index, start : start + length]
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_context, _ = self.bigru(packed)
        compact_context, _ = nn.utils.rnn.pad_packed_sequence(
            packed_context, batch_first=True, total_length=self.config.window_length
        )
        context = motif.new_zeros(
            motif.shape[0], self.config.window_length, self.config.bigru_dimension
        )
        for index in range(motif.shape[0]):
            length = int(lengths[index])
            start = int(left_padding[index])
            context[index, start : start + length] = compact_context[index, :length]
        context = context * mask.unsqueeze(-1).to(context.dtype)
        projected = self.bigru_projection(context)
        projected = self.bigru_residual_norm(motif + self.bigru_dropout(projected))
        projected = projected * mask.unsqueeze(-1).to(projected.dtype)
        return context, projected

    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        mask = attention_mask.bool()
        motif = self.token_cnn(token_ids, mask)
        context, projected = self._contextualize_motifs(motif, mask)
        hidden, attentions = self._run_transformer(projected, mask, return_attention)
        return {
            "motif_tokens": motif,
            "bigru_context_tokens": context,
            "bigru_projected_tokens": projected,
            "transformer_tokens": hidden,
            "center_token": hidden[:, self.config.center_index],
            "attentions": attentions,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
