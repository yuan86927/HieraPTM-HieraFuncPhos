from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
MODEL_SOURCE = REPO_ROOT / "HieraPTM" / "code" / "src"
ARCHIVED_SCORE_TOLERANCE = 1e-5
INPUT_KEYS = (
    "token_ids", "attention_mask", "esm_embeddings", "esm_present_mask",
    "node_esm", "node_positions", "node_coordinates", "node_geometry", "node_mask",
)
CHECKPOINT_SHA256 = {
    "phosphorylation_st": "2ebf4e2a3f53f96c0cf080388b2393d3b7251cd86496882b51ac0dc31bd1073d",
    "phosphorylation_y": "aa1581ba10becb27beebeb0358ae6a34ab68340ca14627159f6abc8bc40b8a30",
}
CASES = {
    "phosphorylation_st": {
        "protein": "AKT1",
        "accession": "P31749",
        "site": "S473",
        "sample_id": "c9a5042e52a3114e60e2cdf03089e539",
        "sequence_61": "TAQMITITPPDQDDSMECVDSERRPHFPQFSYSASGTAXXXXXXXXXXXXXXXXXXXXXXX",
        "split": "relaxed_test",
        "benchmark_label": 1,
        "archived_score": 0.999052226543,
        "evidence": "https://pubmed.ncbi.nlm.nih.gov/15718470/",
    },
    "phosphorylation_y": {
        "protein": "ERK1",
        "accession": "P27361",
        "site": "Y204",
        "sample_id": "10645f21b5ca4489b26f2a533a84b41c",
        "sequence_61": "INTTCDLKICDFGLARIADPEHDHTGFLTEYVATRWYRAPEIMLNSKGYTKSIDIWSVGCI",
        "split": "relaxed_test",
        "benchmark_label": 1,
        "archived_score": 0.982504189014,
        "evidence": "https://pubmed.ncbi.nlm.nih.gov/18983981/",
    },
}


def load_inference_state(path: Path, head: str):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA256[head]:
            raise ValueError(f"Unexpected SHA-256 for released checkpoint {path}")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=False)["model_state"]

    if state["fusion_gamma"].item() != 0:
        raise ValueError("Expected a disabled late-fusion branch.")
    unused_prefixes = ("graph_pool_score.", "graph_summary.", "fusion_gate.")
    return {
        key: value for key, value in state.items()
        if key != "fusion_gamma" and not key.startswith(unused_prefixes)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the archived AKT1 S473 and ERK1 Y204 test inputs on a CPU."
    )
    parser.add_argument("--head", choices=CASES, help="Run one head. The default runs both.")
    parser.add_argument("--checkpoint", type=Path, help="Override best.pt for a single --head.")
    parser.add_argument("--output", type=Path, default=Path("demo_predictions.tsv"))
    args = parser.parse_args()
    start = time.perf_counter()

    if args.checkpoint is not None and args.head is None:
        parser.error("--checkpoint requires --head.")
    if not (MODEL_SOURCE / "models.py").is_file():
        parser.error("Place demo.py in the demo directory at the repository root.")
    sys.path.insert(0, str(MODEL_SOURCE))
    from models import HieraPTM

    torch.set_num_threads(1)
    rows = []
    feature_path = DEMO_DIR / "demo_features.npz"
    if not feature_path.is_file():
        parser.error(f"Missing demo inputs: {feature_path}")
    with np.load(feature_path, allow_pickle=False) as features:
        for head in (args.head,) if args.head else CASES:
            checkpoint = args.checkpoint or REPO_ROOT / "HieraPTM" / "weights" / head / "best.pt"
            if not checkpoint.is_file():
                parser.error(f"Released checkpoint file not found: {checkpoint}")
            model = HieraPTM()
            model.load_state_dict(load_inference_state(checkpoint, head), strict=True)
            model.eval()
            inputs = {key: torch.from_numpy(features[f"{head}.{key}"]) for key in INPUT_KEYS}
            with torch.inference_mode():
                score = torch.sigmoid(model(**inputs)).item()

            case = CASES[head]
            difference = abs(score - case["archived_score"])
            if not 0 <= score <= 1 or difference > ARCHIVED_SCORE_TOLERANCE:
                raise RuntimeError(
                    f"{case['protein']} {case['site']} differs from the archived result "
                    f"(CPU {score:.8f}, archived {case['archived_score']:.8f})."
                )
            rows.append({"head": head, **case, "model_score": f"{score:.8f}", "absolute_difference": f"{difference:.3g}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['protein']} {row['site']}: {row['model_score']} (archived {row['archived_score']:.8f})")
    print(f"Wrote {args.output} in {time.perf_counter() - start:.2f} s.")


if __name__ == "__main__":
    main()
