#!/usr/bin/env python3
"""
scripts/train_minedojo_skill.py  —  Train a single PPO skill for MineDojo.

Trains one SB3 PPO model to acquire a specific Minecraft item (DIA variable).
Must be run once per skill (9 total) before running transfer_minecraft3d_dia.py.

The model is saved to models/minedojo/skill_{var_name}.zip.

Architecture
------------
  MineDojo env (open-ended) → MinedojoObsWrapper → ItemRewardWrapper
  SB3 PPO (MultiInputPolicy: CNN for RGB + MLP for inventory [+ MLP for clip_goal])

Reward shaping (ItemRewardWrapper):
  +1.0       on acquiring target item (var goes 0→1)
  +0.1       each step item remains acquired (sustain)
  -0.005     per step (efficiency pressure)
  +clip_weight * cos_sim(frame, text_embed)   (if --use_clip; default weight 0.1)

CLIP conditioning (--use_clip)
------------------------------
  Each skill has a hardcoded natural-language description in clip_skill.SKILL_TEXTS.
  The CLIP text embedding (512-dim) is added to the obs dict as "clip_goal".
  SB3 MultiInputPolicy picks it up as a third input branch automatically.

  At inference time (transfer_minecraft3d_dia.py), the VLM generates a scene-
  specific instruction and replaces this embedding via option.set_goal_embedding().
  Because CLIP text embeddings live in a shared semantic space, the policy
  generalises from the training descriptions to VLM-generated descriptions.

Usage
-----
  # Train wood-gathering skill WITH CLIP (recommended):
  conda run -n dia-minecraft python scripts/train_minedojo_skill.py \\
      --var wood --steps 500_000 --use_clip

  # Without CLIP (baseline, faster startup):
  conda run -n dia-minecraft python scripts/train_minedojo_skill.py \\
      --var wood --steps 500_000

  # Train all 9 skills in parallel with CLIP:
  for var in wood stone coal ironore furnace stonepickaxe iron ironpickaxe diamond; do
    conda run -n dia-minecraft python scripts/train_minedojo_skill.py \\
        --var $var --use_clip &
  done

Recommended training order (simpler skills first):
  1. wood        (~200k steps, easy: chop trees)
  2. stone       (~300k steps, easy: mine stone)
  3. coal        (~300k steps, easy: mine coal)
  4. ironore     (~600k steps: needs stone_pickaxe first, env provides it in state)
  5. furnace     (~400k steps: craft from stone)
  6. stonepickaxe (~400k steps: craft from wood+stone)
  7. iron        (~800k steps: smelt with furnace+ore+coal)
  8. ironpickaxe (~800k steps: craft from iron+wood)
  9. diamond     (~1M+ steps: mine with iron_pickaxe)
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import minedojo
    MINEDOJO_OK = True
except ImportError:
    MINEDOJO_OK = False
    print("WARNING: minedojo not installed. Install with: pip install minedojo")
    print("         See: https://github.com/MineDojo/MineDojo")

try:
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    SB3_OK = True
except ImportError:
    SB3_OK = False
    print("WARNING: stable-baselines3 not installed. Install with: pip install stable-baselines3")

from dia.evgs_minedojo import make_minedojo_evgs, VAR_NAMES
from dia.options_minedojo import MinedojoObsWrapper, ItemRewardWrapper
from dia.clip_skill import CLIPGoalEncoder, SKILL_TEXTS


# ── Variable ordering (matches 2D Minecraft) ─────────────────────────────────
_VAR_IDX = {n: i for i, n in enumerate(VAR_NAMES)}

# ── Recommended training steps per skill (for reference) ─────────────────────
_RECOMMENDED_STEPS = {
    "wood":         500_000,
    "stone":        500_000,
    "coal":         500_000,
    "ironore":      800_000,
    "furnace":      600_000,
    "stonepickaxe": 600_000,
    "iron":       1_000_000,
    "ironpickaxe":1_000_000,
    "diamond":    1_500_000,
}

# ── MineDojo task IDs for each skill ─────────────────────────────────────────
# Using open-ended task so the agent can gather in any order.
# Skill-specific task IDs can also be used for denser environments:
#   "harvest_log_with_iron_axe-easy", "harvest_cobblestone_with_stone_pickaxe-easy", etc.
_MINEDOJO_TASK = "open-ended"

# ── Environment image size ────────────────────────────────────────────────────
_IMAGE_H, _IMAGE_W = 160, 256


def make_env(var_name: str, seed: int = 0,
             clip_encoder=None, clip_weight: float = 0.1):
    """
    Create a MineDojo env wrapped for skill training.

    If clip_encoder is provided, the CLIP text embedding for var_name is baked
    into the obs wrapper (as 'clip_goal') and used for dense reward shaping.
    """
    if not MINEDOJO_OK:
        raise RuntimeError("minedojo not installed")

    var_idx    = _VAR_IDX[var_name]
    evgs       = make_minedojo_evgs()
    goal_embed = clip_encoder.skill_embedding(var_name) if clip_encoder else None

    def _make():
        raw_env = minedojo.make(
            task_id=_MINEDOJO_TASK,
            image_size=(_IMAGE_H, _IMAGE_W),
            world_seed=seed,
            generate_world_type="default",
        )
        wrapped = MinedojoObsWrapper(raw_env, evgs, goal_embedding=goal_embed)
        shaped  = ItemRewardWrapper(
            wrapped, target_var_idx=var_idx,
            clip_encoder=clip_encoder, clip_weight=clip_weight,
        )
        return shaped

    return _make


def main():
    ap = argparse.ArgumentParser(
        description="Train a PPO skill for a single MineDojo item acquisition.")
    ap.add_argument("--var",     type=str, required=True, choices=VAR_NAMES,
                    help="Target DIA variable / Minecraft item to acquire")
    ap.add_argument("--steps",   type=int, default=None,
                    help="Total training steps (default: recommended per-skill value)")
    ap.add_argument("--outdir",  type=str, default="models/minedojo",
                    help="Directory to save trained model")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--n_envs",  type=int, default=4,
                    help="Number of parallel envs for training")
    ap.add_argument("--lr",      type=float, default=3e-4)
    ap.add_argument("--n_steps", type=int, default=2048,
                    help="Steps per PPO rollout (per env)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--n_epochs",   type=int, default=10)
    ap.add_argument("--ent_coef",   type=float, default=0.01)
    ap.add_argument("--checkpoint_every", type=int, default=50_000,
                    help="Save checkpoint every N steps")
    ap.add_argument("--use_clip",   action="store_true",
                    help="Add CLIP text embedding to obs ('clip_goal') and use "
                         "cosine-similarity dense reward during training. "
                         "Recommended: substantially improves sample efficiency.")
    ap.add_argument("--clip_weight", type=float, default=0.1,
                    help="Weight for CLIP dense reward term (default 0.1)")
    ap.add_argument("--clip_device", type=str, default="cpu",
                    help="Device for CLIP model: 'cpu' or 'cuda' (default cpu)")
    args = ap.parse_args()

    if not MINEDOJO_OK:
        print("ERROR: minedojo not installed. Cannot train.")
        print("  pip install minedojo")
        return
    if not SB3_OK:
        print("ERROR: stable-baselines3 not installed. Cannot train.")
        print("  pip install stable-baselines3[extra]")
        return

    total_steps = args.steps or _RECOMMENDED_STEPS.get(args.var, 1_000_000)
    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, f"skill_{args.var}")

    # ── CLIP setup ────────────────────────────────────────────────────────────
    clip_encoder = None
    if args.use_clip:
        print(f"Loading CLIP (ViT-B/32) on {args.clip_device}...")
        clip_encoder = CLIPGoalEncoder.get(device=args.clip_device)
        goal_text = SKILL_TEXTS.get(args.var, f"gathering {args.var} in Minecraft")
        print(f"  Goal text: \"{goal_text}\"")

    print("=" * 65)
    print(f"  MineDojo PPO Skill Training")
    print(f"  Target:     {args.var}  (var_idx={_VAR_IDX[args.var]})")
    print(f"  Steps:      {total_steps:,}")
    print(f"  n_envs:     {args.n_envs}")
    print(f"  CLIP:       {'enabled (weight=' + str(args.clip_weight) + ')' if args.use_clip else 'disabled'}")
    print(f"  Save path:  {save_path}.zip")
    print("=" * 65)

    # ── Vectorised env ────────────────────────────────────────────────────────
    vec_env = DummyVecEnv([
        make_env(args.var, seed=args.seed + i,
                 clip_encoder=clip_encoder, clip_weight=args.clip_weight)
        for i in range(args.n_envs)
    ])
    vec_env = VecMonitor(vec_env)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // args.n_envs),
        save_path=os.path.join(args.outdir, "checkpoints", args.var),
        name_prefix=f"skill_{args.var}",
    )

    # ── PPO model (MultiInputPolicy handles Dict obs: RGB + inventory) ────────
    model = SB3PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
        # CnnPolicy kwargs for the image branch
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        ),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining {args.var} skill for {total_steps:,} steps...")
    model.learn(
        total_timesteps=total_steps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(save_path)
    print(f"\nSaved: {save_path}.zip")
    print("\nAll 9 skills needed for Phase 2 transfer:")
    for var in VAR_NAMES:
        p    = os.path.join(args.outdir, f"skill_{var}.zip")
        mark = "\u2713" if os.path.exists(p) else " "
        recs = _RECOMMENDED_STEPS.get(var, 1_000_000)
        print(f"  [{mark}] {var:<14}  {p}  (rec: {recs:,} steps)")


if __name__ == "__main__":
    main()
