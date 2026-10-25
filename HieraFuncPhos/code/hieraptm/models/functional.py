"""STY-joint functional transfer from full Task1 G4/CA2-C checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .graph_encoder import build_graph_variant
from .lora_layers import ZeroInitLoRALinear
from .lora_layers import QVLoRALinear


class FullTask1LoRABranch(nn.Module):
    def __init__(self, checkpoint_path: Path, task: str, lora_dropout: float) -> None:
        super().__init__()
        if task not in ("ST", "Y"):
            raise ValueError("Task must be ST or Y")
        self.task = task
        path = Path(checkpoint_path).resolve()
        expected = "phosphorylation_st" if task == "ST" else "phosphorylation_y"
        if expected not in [part.lower() for part in path.parts]:
            raise RuntimeError(f"{task} requires matched {expected} checkpoint")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if len(checkpoint["model_state"]) != 158:
            raise RuntimeError("Expected exactly 158 tensors in full Task1 G4 checkpoint")
        self.encoder = build_graph_variant("G4")
        self.encoder.load_state_dict(checkpoint["model_state"], strict=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for block in self.encoder.parent.backbone.blocks:
            block.attention.qkv = QVLoRALinear(
                block.attention.qkv, rank=8, alpha=16, dropout=lora_dropout
            )
            block.attention.output = ZeroInitLoRALinear(
                block.attention.output, rank=8, dropout=lora_dropout
            )
            block.ffn[0] = ZeroInitLoRALinear(block.ffn[0], rank=4, dropout=lora_dropout)
            block.ffn[3] = ZeroInitLoRALinear(block.ffn[3], rank=4, dropout=lora_dropout)
        for parameter in self.encoder.parent.gate.parameters():
            parameter.requires_grad_(True)
        self.encoder.cross_gamma.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # Preserve frozen Task1 graph and sequence behavior.
        for block in self.encoder.parent.backbone.blocks:
            block.attention.qkv.drop.train(mode)
            block.attention.output.drop.train(mode)
            block.ffn[0].drop.train(mode)
            block.ffn[3].drop.train(mode)
        return self

    def forward(self, token_ids, attention_mask, esm_embeddings, esm_present_mask,
                node_esm, node_positions, node_coordinates, node_geometry, node_mask):
        return self.encoder.encode(
            token_ids, attention_mask, esm_embeddings, esm_present_mask,
            node_esm, node_positions, node_coordinates, node_geometry, node_mask,
        )


class HieraPTMFullJointLoRAV2(nn.Module):
    """Two residue-matched full Task1 branches, one shared functional reader."""

    def __init__(self, task1_st: Path, task1_y: Path,
                 dropout: float = 0.30, lora_dropout: float = 0.05,
                 head_width: int = 128) -> None:
        super().__init__()
        self.st = FullTask1LoRABranch(task1_st, "ST", lora_dropout)
        self.y = FullTask1LoRABranch(task1_y, "Y", lora_dropout)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(2560), nn.Linear(2560, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 192),
        )
        self.global_gate = nn.Linear(384, 1)
        nn.init.zeros_(self.global_gate.weight)
        nn.init.constant_(self.global_gate.bias, -2.0)
        self.functional_adapter = nn.Sequential(
            nn.LayerNorm(192), nn.Linear(192, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 192),
        )
        nn.init.zeros_(self.functional_adapter[-1].weight)
        nn.init.zeros_(self.functional_adapter[-1].bias)
        self.functional_head = nn.Sequential(
            nn.LayerNorm(192), nn.Linear(192, head_width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_width, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.st.train(mode)
        self.y.train(mode)
        self.global_projection.train(mode)
        self.global_gate.train(mode)
        self.functional_adapter.train(mode)
        self.functional_head.train(mode)
        return self

    def forward(self, token_ids, attention_mask, esm_embeddings, esm_present_mask,
                node_esm, node_positions, node_coordinates, node_geometry, node_mask,
                global_embedding, residue_id):
        if residue_id.ndim != 1 or not torch.all((residue_id >= 0) & (residue_id <= 2)):
            raise ValueError("Expected residue_id 0=S, 1=T, 2=Y")
        if global_embedding.ndim != 2 or global_embedding.shape[1] != 2560:
            raise ValueError("Expected frozen protein-global ESM-2 3B [batch,2560]")
        batch = len(residue_id)
        representation = esm_embeddings.new_empty((batch, 192), dtype=torch.float32)
        esm_gate = esm_embeddings.new_empty(batch, dtype=torch.float32)
        for selected, branch in ((residue_id != 2, self.st), (residue_id == 2, self.y)):
            positions = torch.nonzero(selected, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            inputs = (
                token_ids, attention_mask, esm_embeddings, esm_present_mask,
                node_esm, node_positions, node_coordinates, node_geometry, node_mask,
            )
            state = branch(*[item.index_select(0, positions) for item in inputs])
            representation = representation.index_copy(
                0, positions, state["center_token"].float()
            )
            esm_gate = esm_gate.index_copy(
                0, positions, state["parent_esm_gate_center_mean"].float()
            )
        global_vector = self.global_projection(global_embedding.float())
        global_weight = torch.sigmoid(
            self.global_gate(torch.cat([representation, global_vector], dim=-1))
        )
        fused = representation + global_weight * global_vector
        functional = fused + self.functional_adapter(fused)
        return {
            "logits": self.functional_head(functional).squeeze(-1),
            "global_gate": global_weight.squeeze(-1),
            "esm_gate": esm_gate,
        }

    def parameter_audit(self):
        counts = {}
        for task, branch in (("ST", self.st), ("Y", self.y)):
            parent = branch.encoder.parent
            counts[f"{task}_transformer_LoRA"] = sum(
                p.numel() for n, p in parent.named_parameters()
                if p.requires_grad and not n.startswith("gate.")
            )
            counts[f"{task}_ESM_gate"] = sum(p.numel() for p in parent.gate.parameters())
            counts[f"{task}_cross_gamma"] = int(branch.encoder.cross_gamma.numel())
        counts["shared_global_projection_and_gate"] = (
            sum(p.numel() for p in self.global_projection.parameters())
            + sum(p.numel() for p in self.global_gate.parameters())
        )
        counts["shared_adapter"] = sum(p.numel() for p in self.functional_adapter.parameters())
        counts["shared_head"] = sum(p.numel() for p in self.functional_head.parameters())
        expected = {
            "ST_transformer_LoRA": 59392, "ST_ESM_gate": 147840,
            "ST_cross_gamma": 1,
            "Y_transformer_LoRA": 59392, "Y_ESM_gate": 147840,
            "Y_cross_gamma": 1,
            "shared_global_projection_and_gate": 181889,
            "shared_adapter": 25216, "shared_head": 25217,
        }
        actual = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if counts != expected or actual != sum(counts.values()):
            raise RuntimeError(f"Full Task1 joint V2 trainable audit failed: {counts}")
        frozen_unexpected = [n for n, p in self.named_parameters() if p.requires_grad and not (
            n.startswith(("global_projection.", "global_gate.",
                          "functional_adapter.", "functional_head.")) or
            n in ("st.encoder.cross_gamma", "y.encoder.cross_gamma") or
            n.startswith(("st.encoder.parent.gate.", "y.encoder.parent.gate.")) or
            (n.startswith(("st.encoder.parent.backbone.blocks.",
                           "y.encoder.parent.backbone.blocks.")) and n.endswith(
                 (".q_a.weight", ".q_b.weight", ".v_a.weight", ".v_b.weight",
                  ".a.weight", ".b.weight")))
        )]
        if frozen_unexpected:
            raise RuntimeError(f"Unexpected trainable full Task1 weights: {frozen_unexpected}")
        return {
            "counts": counts, "trainable_parameters": actual,
            "full_Task1_graph_in_forward": True,
            "ST_and_Y_matched_checkpoints": True,
            "STY_joint_training": True,
            "graph_original_weights_frozen_except_cross_gamma": True,
        }
