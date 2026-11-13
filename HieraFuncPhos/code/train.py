"""Train the final HieraPTM pipeline on fixed public splits; never score test."""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("config.json"))
    parser.add_argument("--rep", type=int, nargs="+", required=True, choices=range(1, 10),
                        help="Public replicate number(s), e.g. --rep 1 or --rep 1 2 3 4 5 6 7 8 9")
    args = parser.parse_args()
    if len(set(args.rep)) != len(args.rep):
        parser.error("Replicates must be unique")
    from hieraptm.pipeline import train
    for replicate in args.rep:
        train(args.config, replicate)


if __name__ == "__main__":
    main()
