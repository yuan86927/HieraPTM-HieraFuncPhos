from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from settings import create_settings, selected_heads
from pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--head', default='all')
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--features-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.set_defaults(stage='evaluate')
    args = parser.parse_args()
    settings = create_settings(args.data_dir, args.features_dir, args.output_dir, args.seed)
    run(settings, selected_heads(settings, args.head), args.stage)


if __name__ == '__main__':
    main()
