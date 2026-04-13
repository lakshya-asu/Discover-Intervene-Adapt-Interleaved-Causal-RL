#!/usr/bin/env python3
"""
Train a voxel-conditioned PPO gather policy for a single DIA skill.

The policy observes:
  target_voxel  (7*3*7=147,) float32 binary -- 1 where voxel contains target block
  agent_state   (6,)          float32        -- [pitch, yaw, health, food, on_ground, has_tool]

And acts over Discrete(10) gather actions (see GATHER_ACTIONS in options_minedojo.py).

Usage
-----
    conda run -n dia-minecraft python scripts/train_gather_skill.py \\
        --skill wood \\
        --steps 500000 \\
        --seed 0 \\
        --out models/minedojo/gather_wood.zip

    # All gather skills back-to-back (same seed):
    for skill in wood stone coal ironore diamond; do
        conda run -n dia-minecraft python scripts/train_gather_skill.py \\
            --skill $skill --steps 500000 --seed 0 \\
            --out models/minedojo/gather_${skill}.zip
    done
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

# Allow importing from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Tool requirements per skill (EVGS variable index, or -1 for bare hand)
# Indices match VAR_NAMES in evgs_minedojo.py:
#   0=wood 1=stone 2=coal 3=ironore 4=furnace 5=stonepickaxe 6=iron 7=ironpickaxe 8=diamond
# ---------------------------------------------------------------------------

_TOOL_VAR = {
    "wood":    -1,   # bare hand (wood is mineable by hand)
    "stone":   -1,   # bare hand (slower but works)
    "coal":    -1,   # bare hand (slower but works)
    "ironore":  5,   # stonepickaxe required
    "diamond":  7,   # ironpickaxe required
}

GATHER_SKILLS = list(_TOOL_VAR.keys())


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO gather policy on MineDojo voxel obs")
    p.add_argument("--skill",  required=True, choices=GATHER_SKILLS,
                   help="gather skill to train")
    p.add_argument("--steps",  type=int, default=500_000,
                   help="total PPO timesteps (default 500000)")
    p.add_argument("--seed",   type=int, default=0,
                   help="world seed for MineDojo (default 0)")
    p.add_argument("--out",    type=str, default=None,
                   help="output path for saved .zip (default: models/minedojo/gather_<skill>.zip)")
    p.add_argument("--voxel_size", type=int, nargs=3, default=[13, 7, 13],
                   metavar=("X", "Y", "Z"),
                   help="voxel grid size (default 13 7 13)")
    p.add_argument("--n_steps",   type=int, default=2048,
                   help="PPO n_steps rollout buffer size (default 2048)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="PPO mini-batch size (default 64)")
    p.add_argument("--lr",    type=float, default=3e-4,
                   help="PPO learning rate (default 3e-4)")
    p.add_argument("--dry_run", action="store_true",
                   help="init env + model, print spaces, then exit without training")
    return p.parse_args()


def make_env(skill: str, seed: int, voxel_size: list):
    """Build and wrap the MineDojo environment for gather skill training."""
    import minedojo
    from src.dia.evgs_minedojo import make_minedojo_evgs
    from src.dia.options_minedojo import VoxelGatherWrapper, VoxelRewardWrapper

    # MineDojo voxel_size must be a dict with xmin/xmax/ymin/ymax/zmin/zmax.
    # Convert (X, Y, Z) tuple to symmetric dict centered on the agent.
    vs = voxel_size
    voxel_dict = dict(
        xmin=-(vs[0] // 2), xmax=vs[0] // 2,
        ymin=-(vs[1] // 2), ymax=vs[1] // 2,
        zmin=-(vs[2] // 2), zmax=vs[2] // 2,
    )
    print(f"[env] making MineDojo open-ended  seed={seed}  biome=forest  voxel_dict={voxel_dict}")
    raw_env = minedojo.make(
        task_id="open-ended",
        image_size=(64, 64),
        world_seed=str(seed),
        specified_biome="forest",
        use_voxel=True,
        voxel_size=voxel_dict,
    )

    evgs = make_minedojo_evgs()
    tool_idx = _TOOL_VAR[skill]

    env = VoxelGatherWrapper(
        raw_env, evgs,
        skill_name=skill,
        tool_var_idx=tool_idx,
        voxel_size=tuple(voxel_size),
    )
    env = VoxelRewardWrapper(env, skill_name=skill, evgs=evgs)
    return env


try:
    import gymnasium as _gymnasium
    _GymEnvBase = _gymnasium.Env
except ImportError:
    import gym as _gymnasium
    _GymEnvBase = _gymnasium.Env


class _GymAdapter(_GymEnvBase):
    """
    Thin adapter to make VoxelRewardWrapper compatible with SB3's env expectations.

    Inherits from gymnasium.Env so that SB3's isinstance check passes.
    """

    metadata = {"render_modes": []}

    def __init__(self, wrapped):
        super().__init__()
        self._w = wrapped
        self.observation_space = wrapped.observation_space
        self.action_space      = wrapped.action_space

    def reset(self, **kwargs):
        # MineDojo doesn't accept 'seed' or 'options' kwargs from gymnasium API
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        obs = self._w.reset(**kwargs)
        return (obs[0] if isinstance(obs, tuple) else obs), {}

    def step(self, action):
        result = self._w.step(action)
        obs, rew, done, info = result
        # gymnasium step → (obs, reward, terminated, truncated, info)
        return obs, float(rew), bool(done), False, info

    def render(self):
        pass

    def close(self):
        try:
            self._w.env.env.close()
        except Exception:
            pass

    # Forward evgs helpers used by VoxelGatherOption
    def get_obs(self):
        return self._w.get_obs()

    def get_raw_obs(self):
        return self._w.get_raw_obs()


def train(args):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("ERROR: stable-baselines3 not installed.  Run: pip install stable-baselines3")
        sys.exit(1)

    # Output path
    out_path = args.out or f"models/minedojo/gather_{args.skill}.zip"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Build env
    raw_wrapped = make_env(args.skill, args.seed, args.voxel_size)
    env = Monitor(_GymAdapter(raw_wrapped))

    print(f"\n[spaces]")
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")

    if args.dry_run:
        print("\n[dry_run] resetting env...")
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        for k, v in obs.items():
            print(f"  {k}: shape={v.shape}  dtype={v.dtype}  "
                  f"min={v.min():.3f}  max={v.max():.3f}  sum={v.sum():.1f}")
        print("[dry_run] OK -- exiting without training")
        env.close()
        return

    # PPO model
    voxel_dim = int(np.prod(args.voxel_size))
    # Wider first layer for larger voxel obs (13x7x13 = 1183 dims)
    hidden = 512 if voxel_dim > 500 else 256
    policy_kwargs = dict(
        net_arch=[hidden, hidden],
    )

    print(f"\n[ppo] building model  skill={args.skill}  voxel_dim={voxel_dim}  hidden={hidden}")
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        device="auto",
        policy_kwargs=policy_kwargs,
    )

    print(f"[ppo] training for {args.steps:,} steps  ->  {out_path}")
    model.learn(
        total_timesteps=args.steps,
        progress_bar=True,
    )

    model.save(out_path)
    print(f"\n[done] model saved to {out_path}")
    env.close()


def main():
    args = parse_args()
    print(f"=== train_gather_skill  skill={args.skill}  seed={args.seed}  "
          f"steps={args.steps:,} ===")
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n[interrupted]")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
