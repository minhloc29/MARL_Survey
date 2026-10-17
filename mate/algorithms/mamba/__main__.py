"""Evaluate a MATE MAMBA checkpoint: python -m mate.algorithms.mamba."""

import argparse
import json

import numpy as np

from .mamba import MAMBATrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    trainer = MAMBATrainer.from_checkpoint(args.checkpoint, device=args.device)
    try:
        stats = [trainer.collect_episode(args.seed + i, training=False) for i in range(args.episodes)]
        print(json.dumps({key: float(np.mean([row[key] for row in stats])) for key in stats[0]}, indent=2))
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()
