"""Score the selected final HieraPTM model on fixed validation or test data."""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("config.json"))
    parser.add_argument("--rep", type=int, nargs="+", required=True, choices=range(1, 10))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    if len(set(args.rep)) != len(args.rep):
        parser.error("Replicates must be unique")
    from hieraptm.evaluation import evaluate
    for replicate in args.rep:
        evaluate(args.config, replicate, args.split)


if __name__ == "__main__":
    main()
