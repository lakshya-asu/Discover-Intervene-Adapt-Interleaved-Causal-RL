#!/usr/bin/env python3
"""
scripts/train_primitive.py — Train one primitive PPO option for MineRL.

Each high-level skill is decomposed into stage-specific PPO options.
The valid stages per skill are defined by SKILL_PRIMITIVE_CHAIN in
primitives_minerl.py — only those stages should be trained for a given var.

Primitive vocabulary
--------------------
  approach      : navigate toward the target block
  aim           : place crosshair on the target block face
  attack        : break the block (gathering: wood/stone/coal/ironore/diamond)
  interact      : right-click open GUI (crafting/smelting skills)
  navigate_deep : descend underground (diamond only)

Skill → valid stages (SKILL_PRIMITIVE_CHAIN)
--------------------------------------------
  wood        : approach aim attack
  stone       : approach aim attack
  coal        : approach aim attack
  ironore     : approach aim attack
  diamond     : approach navigate_deep aim attack
  furnace     : approach aim interact
  stonepickaxe: approach aim interact
  iron        : approach aim interact
  ironpickaxe : approach aim interact

Usage
-----
  # Wood (3 stages):
  for stage in approach aim attack; do
    conda run -n dia-minecraft python scripts/train_primitive.py \\
        --var wood --stage $stage
  done

  # Diamond (4 stages, navigate_deep is diamond-specific):
  for stage in approach navigate_deep aim attack; do
    conda run -n dia-minecraft python scripts/train_primitive.py \\
        --var diamond --stage $stage
  done

  # Quick smoke test:
  conda run -n dia-minecraft python scripts/train_primitive.py \\
      --var wood --stage approach --steps 100

Output
------
  models/minerl/approach_wood.zip
  models/minerl/aim_wood.zip
  models/minerl/attack_wood.zip
  ... etc.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import gym
    import minerl  # noqa: registers MineRL envs
    MINERL_OK = True
except ImportError:
    MINERL_OK = False
    print("WARNING: minerl not installed.")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    SB3_OK = True
except ImportError:
    SB3_OK = False
    print("WARNING: stable-baselines3 not installed.")

from dia.evgs_minerl import make_minerl_evgs
from dia.clip_skill import CLIPGoalEncoder
from dia.options_minerl import (
    MineRLObsWrapper, BiasedDiscrete, N_ACTIONS,
)
from dia.primitives_minerl import (
    ALL_STAGES, SKILL_PRIMITIVE_CHAIN, PRIMITIVE_TEXTS,
    STAGE_PREFERRED_ACTIONS, STAGE_MAX_STEPS, STAGE_CLIP_WEIGHT,
    make_primitive_wrapper, validate_stage_for_var,
    primitive_model_path, print_primitive_status,
)

_MINERL_ENV_ID = "MineRLObtainDiamondShovel-v0"

VAR_NAMES = [
    "wood", "stone", "coal", "ironore", "furnace",
    "stonepickaxe", "iron", "ironpickaxe", "diamond",
]

# Recommended training steps per stage (can be overridden with --steps)
_RECOMMENDED_STEPS: dict = {
    "approach":      200_000,
    "navigate_deep": 150_000,
    "aim":           100_000,
    "attack":        300_000,
    "interact":      200_000,
}


def make_env(var_name: str, stage: str, seed: int = 0,
             clip_encoder=None, explore_bias: float = 0.7):
    """
    Create a MineRL env for training a single primitive stage.

    Stack:
      MineRLObsWrapper  (rgb + inventory + clip_goal)
      └── {Approach|Aim|Attack|Interact|NavigateDeep}RewardWrapper
    """
    var_idx     = VAR_NAMES.index(var_name)
    evgs        = make_minerl_evgs()
    clip_weight = STAGE_CLIP_WEIGHT.get(stage, 0.3)

    # CLIP text embedding for this stage
    goal_embed = None
    if clip_encoder is not None:
        text = (PRIMITIVE_TEXTS.get(stage, {})
                .get(var_name, f"acquiring {var_name} in Minecraft"))
        goal_embed = clip_encoder.encode_text(text)
        print(f"  Goal text: \"{text}\"")

    raw_env = gym.make(_MINERL_ENV_ID)
    raw_env.seed(seed)

    env = MineRLObsWrapper(raw_env, evgs, goal_embedding=goal_embed)
    env = make_primitive_wrapper(
        stage=stage, env=env, target_var_idx=var_idx,
        clip_encoder=clip_encoder, clip_weight=clip_weight,
    )

    # Bias action sampling toward stage-relevant actions during PPO rollouts
    if explore_bias > 0 and BiasedDiscrete is not None:
        preferred = STAGE_PREFERRED_ACTIONS.get(stage, list(range(N_ACTIONS)))
        env.action_space = BiasedDiscrete(N_ACTIONS, preferred,
                                           bias_prob=explore_bias)
        print(f"  [BiasedDiscrete] preferred={preferred}")

    return env


def main():
    ap = argparse.ArgumentParser(
        description="Train one PPO primitive (approach/aim/attack/interact/navigate_deep).")
    ap.add_argument("--var",   required=True, choices=VAR_NAMES,
                    help="Target DIA variable / Minecraft item")
    ap.add_argument("--stage", required=True, choices=list(ALL_STAGES),
                    help="Which primitive stage to train")
    ap.add_argument("--steps", type=int, default=0,
                    help="Training steps (0 = recommended default)")
    ap.add_argument("--outdir", type=str, default="models/minerl")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--device",       type=str,   default="auto")
    ap.add_argument("--explore_bias", type=float, default=0.7)
    ap.add_argument("--no_clip",      action="store_true",
                    help="Disable CLIP (faster startup, lower quality)")
    ap.add_argument("--clip_device",  type=str, default="cpu")
    args = ap.parse_args()

    # Validate that this stage is valid for this var
    if not validate_stage_for_var(args.var, args.stage):
        valid = SKILL_PRIMITIVE_CHAIN.get(args.var, [])
        print(f"ERROR: stage '{args.stage}' is not valid for var '{args.var}'.")
        print(f"  Valid stages for '{args.var}': {valid}")
        return

    if not MINERL_OK:
        print("ERROR: minerl not installed.")
        return
    if not SB3_OK:
        print("ERROR: stable-baselines3 not installed.")
        return

    # ── CLIP setup ────────────────────────────────────────────────────────────
    clip_encoder = None
    if not args.no_clip:
        print(f"Loading CLIP (ViT-B/32) on {args.clip_device}...")
        clip_encoder = CLIPGoalEncoder.get(device=args.clip_device)

    n_steps  = args.steps if args.steps > 0 else _RECOMMENDED_STEPS.get(args.stage, 200_000)
    out_path = primitive_model_path(args.outdir, args.var, args.stage)
    ckpt_dir = os.path.join(args.outdir, "checkpoints", args.stage, args.var)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    chain = SKILL_PRIMITIVE_CHAIN.get(args.var, [])
    print("=" * 65)
    print(f"  Primitive PPO Training")
    print(f"  Variable:   {args.var}")
    print(f"  Stage:      {args.stage}  (chain: {' → '.join(chain)})")
    print(f"  Env:        {_MINERL_ENV_ID}")
    print(f"  Steps:      {n_steps:,}")
    print(f"  CLIP:       {'enabled' if clip_encoder else 'disabled'}")
    print(f"  Output:     {out_path}")
    print("=" * 65)

    env = make_env(args.var, args.stage, seed=args.seed,
                   clip_encoder=clip_encoder, explore_bias=args.explore_bias)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(n_steps // 10, 5_000),
        save_path=ckpt_dir,
        name_prefix=f"ckpt_{args.stage}_{args.var}",
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
        ent_coef=0.02,
        tensorboard_log=os.path.join(args.outdir, "tb_logs"),
    )

    print(f"\nStarting training...")
    model.learn(total_timesteps=n_steps, callback=checkpoint_cb, progress_bar=True)

    model.save(out_path)
    print(f"\nSaved: {out_path}")
    env.close()

    # Show status of all primitives for this var
    print_primitive_status([args.var], model_dir=args.outdir)


if __name__ == "__main__":
    main()
