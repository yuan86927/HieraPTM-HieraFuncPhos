"""Task1 train command; implementation lives in src/."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from configuration import load_config, resolve, selected_heads
from pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task1.json")
    parser.add_argument("--head", default="all")
    parser.add_argument("--stage", choices=("train", "check-data", "check-inputs"), default="train")
    args = parser.parse_args()
    config = load_config(resolve(args.config))
    run(config, selected_heads(config, args.head), args.stage)


if __name__ == "__main__":
    main()
