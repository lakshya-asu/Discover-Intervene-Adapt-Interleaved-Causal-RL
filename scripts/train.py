"""CLI wrapper for training PPO agents."""

from __future__ import annotations

import argparse
from typing import Optional

from dia.agents.ppo_sb3 import train as train_ppo


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO using Stable-Baselines3")
    parser.add_argument(
        "--env",
        required=True,
        help="Gym ID or module:Callable path for the training environment",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Number of training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    train_ppo(env_name=args.env, total_steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
