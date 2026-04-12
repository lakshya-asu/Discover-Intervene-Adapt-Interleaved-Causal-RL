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

from dia.clip_skill import CLIPGoalEncoder, SKILL_TEXTS

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


def make_env(var_name: str, seed: int = 0, explore_bias: float = 0.0,
             clip_encoder=None, clip_weight: float = 0.1):
    """Create a wrapped MineRL env for training skill_{var_name}.

    Args:
        explore_bias: probability (0–1) of sampling from skill-preferred actions
            during rollout collection. 0 = uniform random (SB3 default).
            ~0.7 is recommended: makes PPO see reward signals much sooner.
        clip_encoder: CLIPGoalEncoder instance (optional). If provided, bakes a
            512-dim CLIP text embedding into obs as 'clip_goal' and adds CLIP
            cosine-similarity dense reward scaled by clip_weight.
    """
    import gym
    import minerl  # noqa: registers envs
    from dia.evgs_minerl import make_minerl_evgs
    from dia.options_minerl import (MineRLObsWrapper, ItemRewardWrapper,
                                    BiasedDiscrete, SKILL_PREFERRED_ACTIONS,
                                    N_ACTIONS)

    var_idx    = VAR_NAMES.index(var_name)
    evgs       = make_minerl_evgs()
    goal_embed = clip_encoder.skill_embedding(var_name) if clip_encoder else None

    raw_env = gym.make(_MINERL_ENV_ID)
    raw_env.seed(seed)

    env = MineRLObsWrapper(raw_env, evgs, goal_embedding=goal_embed)
    env = ItemRewardWrapper(env, target_var_idx=var_idx,
                            clip_encoder=clip_encoder, clip_weight=clip_weight)

    # Bias action sampling toward skill-relevant actions during rollout collection.
    if explore_bias > 0 and BiasedDiscrete is not None:
        preferred = SKILL_PREFERRED_ACTIONS.get(var_name, list(range(N_ACTIONS)))
        env.action_space = BiasedDiscrete(N_ACTIONS, preferred, bias_prob=explore_bias)
        print(f"  [explore_bias={explore_bias:.2f}] preferred actions for {var_name}: {preferred}")

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
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--device",       type=str,   default="auto",
                    help="'cpu', 'cuda', or 'auto'")
    ap.add_argument("--explore_bias", type=float, default=0.7,
                    help="Probability of sampling skill-preferred actions during "
                         "PPO rollout collection (0=uniform random, 0.7=recommended). "
                         "Makes PPO see reward much sooner on hard-exploration tasks.")
    ap.add_argument("--use_clip",    action="store_true",
                    help="Add CLIP text embedding ('clip_goal') to obs and add "
                         "cosine-similarity dense reward. Recommended.")
    ap.add_argument("--clip_weight", type=float, default=0.1,
                    help="Weight for CLIP reward term (default 0.1)")
    ap.add_argument("--clip_device", type=str, default="cpu",
                    help="Device for CLIP: 'cpu' or 'cuda'")
    args = ap.parse_args()

    # ── CLIP setup ────────────────────────────────────────────────────────────
    clip_encoder = None
    if args.use_clip:
        print(f"Loading CLIP (ViT-B/32) on {args.clip_device}...")
        clip_encoder = CLIPGoalEncoder.get(device=args.clip_device)
        goal_text = SKILL_TEXTS.get(args.var, f"gathering {args.var} in Minecraft")
        print(f"  Goal text: \"{goal_text}\"")

    n_steps = args.steps if args.steps > 0 else _RECOMMENDED.get(args.var, 500_000)
    out_path = os.path.join(args.outdir, f"skill_{args.var}.zip")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training PPO skill for variable: {args.var}")
    print(f"  env:          {_MINERL_ENV_ID}")
    print(f"  steps:        {n_steps:,}")
    print(f"  output:       {out_path}")
    print(f"  device:       {args.device}")
    print(f"  explore_bias: {args.explore_bias:.2f}")
    print(f"  CLIP:         {'enabled (weight=' + str(args.clip_weight) + ')' if args.use_clip else 'disabled'}")

    # ── Build env ─────────────────────────────────────────────────────────────
    env = make_env(args.var, seed=args.seed, explore_bias=args.explore_bias,
                   clip_encoder=clip_encoder, clip_weight=args.clip_weight)

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
