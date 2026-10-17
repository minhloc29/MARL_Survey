"""Evaluate with: python -m mate.algorithms.aorpo --checkpoint PATH."""

import argparse
import json

import numpy as np

from .aorpo import AORPOTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=10000, help="Offset added to the saved training seed")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    trainer = AORPOTrainer.from_checkpoint(args.checkpoint, args.device)
    try:
        rows = [trainer.collect_episode(args.seed + i, training=False) for i in range(args.episodes)]
        print(json.dumps({key: float(np.mean([row[key] for row in rows])) for key in rows[0]}, indent=2))
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()
