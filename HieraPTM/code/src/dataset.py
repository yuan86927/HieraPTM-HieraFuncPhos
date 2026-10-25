"""Read self-contained final split tables; no resampling or split overlays."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


VOCABULARY = ("<PAD>", "X", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
              "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "B", "Z", "J", "U", "O")
SPLIT_CODE = {"fit": 0, "development": 1, "relaxed_test": 2}
METADATA = ("sample_id", "protein_id", "ptm", "head", "sequence_61", "split", "center_aa",
            "center_position_1based", "outer_source_split", "outer_source_row", "esm_available", "baseline_score")


class FinalSplitData:
    def __init__(self, path: Path, head: str, specification: dict) -> None:
        columns = [*METADATA, "label", "fold_id", "valid_mask_61"]
        frame = pd.read_csv(path, sep="\t", usecols=columns, dtype={"valid_mask_61": str})
        if len(frame) != specification["expected_rows"] or not frame.sample_id.is_unique:
            raise ValueError(f"Wrong sample count or duplicate sample ID: {head}")
        if set(frame["head"]) != {head} or set(frame["label"]) != {0, 1}:
            raise ValueError(f"Wrong head or labels: {head}")
        codes = frame["split"].map(SPLIT_CODE)
        if codes.isna().any() or set(codes) != {0, 1, 2}:
            raise ValueError(f"Invalid split names: {head}")
        self.fold_ids = codes.to_numpy(dtype=np.uint8)
        if not np.array_equal(self.fold_ids, frame["fold_id"].to_numpy()):
            raise ValueError(f"split and fold_id disagree: {head}")
        if "split_code_sha256" in specification:
            observed = hashlib.sha256(self.fold_ids.tobytes()).hexdigest()
            if observed != specification["split_code_sha256"]:
                raise ValueError(f"Final split membership/order differs: {head}")
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        sequence = frame["sequence_61"].astype(str)
        masks = frame["valid_mask_61"].astype(str)
        if not sequence.str.len().eq(61).all() or not masks.str.fullmatch("[01]{61}").all():
            raise ValueError(f"Invalid 61-residue window/mask: {head}")
        encoded = np.frombuffer("".join(sequence).encode("ascii"), dtype=np.uint8).reshape(-1, 61)
        lookup = np.full(256, 255, dtype=np.uint8)
        for index, residue in enumerate(VOCABULARY):
            if len(residue) == 1:
                lookup[ord(residue)] = index
        self.token_ids = lookup[encoded]
        if np.any(self.token_ids == 255):
            raise ValueError(f"Unknown residue in sequence: {head}")
        self.attention_masks = np.frombuffer("".join(masks).encode("ascii"), dtype=np.uint8).reshape(-1, 61) == ord("1")
        if not self.attention_masks[:, 30].all():
            raise ValueError(f"Invalid center mask: {head}")
        self.metadata = frame[list(METADATA)].fillna("").astype(str).to_dict("records")

    def counts(self) -> dict[str, int]:
        return {name: int(np.count_nonzero(self.fold_ids == code)) for name, code in SPLIT_CODE.items()}
