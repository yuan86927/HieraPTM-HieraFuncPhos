"""Configuration and package-relative paths shared by train/evaluate."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol"]["training_seed"] != 20260830:
        raise ValueError("Final training seed must remain 20260830")
    if set(config["head_order"]) != set(config["heads"]):
        raise ValueError("Head definitions/order disagree")
    return config


def selected_heads(config: dict, head: str) -> list[str]:
    heads = config["head_order"] if head == "all" else [head]
    if any(value not in config["heads"] for value in heads):
        raise ValueError("Unknown PTM head")
    return heads
