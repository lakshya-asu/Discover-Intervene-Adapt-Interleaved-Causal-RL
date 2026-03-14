#!/usr/bin/env python3
"""
scripts/train_minerl_skill.py  —  Train one PPO skill for MineRL 3D Minecraft.

Trains a single SB3 PPO agent to acquire one DIA variable in MineRL.
The skill is saved as models/minerl/skill_{var}.zip for use in Phase 2.

Usage
-----
  # Train all skills (one at a time, takes hours each):
  conda run -n dia-minecraft python scripts/train_minerl_skill.py --var wood
  conda run -n dia-minecraft python scripts/train_minerl_skill.py --var stone
  conda run -n dia-minecraft python scripts/train_minerl_skill.py --var coal

  # Quick smoke test (100 steps, just checks code runs):
  conda run -n dia-minecraft python scripts/train_minerl_skill.py --var wood --steps 100

Recommended training steps (rough estimates, highly task-dependent):
  wood      : 500,000   (chop tree — relatively easy)
  stone     : 500,000   (mine cobblestone — needs stone_pickaxe or fists)
  coal      : 500,000   (mine coal ore — needs pickaxe)
  ironore   : 1,000,000 (mine iron ore — needs stone_pickaxe)
  furnace   : 500,000   (craft furnace — needs wood+stone)
  stonepick : 800,000   (craft stone_pickaxe — needs wood+stone)
  iron      : 1,000,000 (smelt iron — needs furnace+coal+ironore)
  ironpick  : 1,000,000 (craft iron_pickaxe — needs iron+wood)
  diamond   : 2,000,000 (mine diamond — needs iron_pickaxe)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

_RECOMMENDED = {
    "wood":        500_000,
    "stone":       500_000,
    "coal":        500_000,
    "ironore":   1_000_000,
    "furnace":     500_000,
    "stonepickaxe": 800_000,
    "iron":      1_000_000,
    "ironpickaxe": 1_000_000,
    "diamond":   2_000_000,
}

VAR_NAMES = ["wood", "stone", "coal", "ironore", "furnace",
             "stonepickaxe", "iron", "ironpickaxe", "diamond"]

# MineRL env used for all skill training  (full tech tree available)
_MINERL_ENV_ID = "MineRLObtainDiamondShovel-v0"


def make_env(var_name: str, seed: int = 0):
    """Create a wrapped MineRL env for training skill_{var_name}."""
    import gym
    import minerl  # noqa: registers envs
    from dia.evgs_minerl import make_minerl_evgs
    from dia.options_minerl import MineRLObsWrapper, ItemRewardWrapper

    var_idx = VAR_NAMES.index(var_name)
    evgs    = make_minerl_evgs()

    raw_env = gym.make(_MINERL_ENV_ID)
    raw_env.seed(seed)

    env = MineRLObsWrapper(raw_env, evgs)
    env = ItemRewardWrapper(env, target_var_idx=var_idx)
    return env


def main():
    ap = argparse.ArgumentParser(
        description="Train one PPO skill option for MineRL 3D Minecraft.")
    ap.add_argument("--var",    required=True, choices=VAR_NAMES,
                    help="Which DIA variable to train a skill for")
    ap.add_argument("--steps",  type=int, default=0,
                    help="Total env steps (0 = use recommended default)")
    ap.add_argument("--outdir", type=str, default="models/minerl",
                    help="Output directory for saved skill")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--device", type=str, default="auto",
                    help="'cpu', 'cuda', or 'auto'")
    args = ap.parse_args()

    n_steps = args.steps if args.steps > 0 else _RECOMMENDED.get(args.var, 500_000)
    out_path = os.path.join(args.outdir, f"skill_{args.var}.zip")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training PPO skill for variable: {args.var}")
    print(f"  env:      {_MINERL_ENV_ID}")
    print(f"  steps:    {n_steps:,}")
    print(f"  output:   {out_path}")
    print(f"  device:   {args.device}")

    # ── Build env ─────────────────────────────────────────────────────────────
    env = make_env(args.var, seed=args.seed)

    # ── Train PPO ─────────────────────────────────────────────────────────────
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    checkpoint_cb = CheckpointCallback(
        save_freq=max(n_steps // 10, 10_000),
        save_path=args.outdir,
        name_prefix=f"ckpt_{args.var}",
        verbose=1,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        device=args.device,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=os.path.join(args.outdir, "tb_logs"),
    )

    print(f"\nStarting training... (this may take a long time)")
    model.learn(total_timesteps=n_steps, callback=checkpoint_cb,
                progress_bar=True)

    model.save(out_path)
    print(f"\nSaved skill model: {out_path}")
    env.close()


if __name__ == "__main__":
    main()
