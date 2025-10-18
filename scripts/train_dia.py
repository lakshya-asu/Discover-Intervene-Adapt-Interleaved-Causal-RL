#!/usr/bin/env python3
# scripts/train_dia.py
"""
Minimal DIA training script wiring EVGS + PCG + SIG + InterventionSelector + Runner.

Features:
- Choose PCG backend: "notears" (DifferentiablePCG), "variational" (VariationalPCG), or "simple".
- Simple EVGS adapter for low-dim continuous states (e.g., CartPole).
- Builds a tiny SIG (two skills) and runs the DIA loop with logging.
- Optional PPO option training for each skill via a shaped-reward wrapper (requires stable-baselines3).

Usage examples:
  python scripts/train_dia.py --env_id CartPole-v1 --pcg notears --steps 200 --logdir runs/dia_demo
  python scripts/train_dia.py --env_id CartPole-v1 --pcg variational --steps 200 --use_ppo_options --train_options
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Callable, Any
import numpy as np

try:
    import gym  # Gym (or Gymnasium if you prefer)
except Exception:
    gym = None

from dia.evgs import EVGS
from dia.types import Subgoal, Predicate
from dia.sig import SIGraph, Skill
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.logging_utils import TBLogger

# PCG backends
from dia.pcg import SimplePCG, PCGConfig  # simple, numpy-based probabilities
from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.pcg_variational import VariationalPCG, VariationalPCGConfig

# Option policies
from dia.options import OptionConfig, RandomOption

# Optional PPO option (only if SB3 installed)
try:
    from dia.options import PPOOption
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False


# -------------------------- EVGS adapter for low-dim states --------------------------

def build_lowdim_evgs(obs_dim: int, names: Optional[List[str]] = None) -> EVGS:
    """Observation -> variable vector (identity mapping for low-dim environments)."""
    if names is None:
        names = [f"X{i}" for i in range(obs_dim)]
    def obs_to_vars(obs):
        arr = np.array(obs, dtype=float).reshape(-1)
        if len(arr) < len(names):
            # pad if env observation is smaller than expected
            pad = np.zeros(len(names))
            pad[:len(arr)] = arr
            return pad
        return arr[:len(names)]
    return EVGS(var_names=names, obs_to_vars=obs_to_vars)


# -------------------------- Optional PPO option shaping wrapper --------------------------

def make_shaped_env(env: Any, evgs: EVGS, subgoal: Subgoal):
    """
    Return a simple reward-shaped environment for training an option:
      +1 (and done=True) when the subgoal predicate holds between successive X.
    """
    import types

    class ShapedEnv(gym.Wrapper):
        def __init__(self, base_env):
            super().__init__(base_env)
            self._last_obs = None
            self._last_x = None

        def reset(self, **kwargs):
            obs = self.env.reset(**kwargs)
            self._last_obs = obs
            self._last_x = evgs.extract(obs)
            return obs

        def step(self, action):
            obs, r, done, info = self.env.step(action)
            x = evgs.extract(obs)
            reward = 0.0
            if self._last_x is not None:
                if EVGS.predicate_holds(self._last_x, x, subgoal):
                    reward = 1.0
                    done = True
            self._last_obs = obs
            self._last_x = x
            # Allow continuing past 'done' if caller wants (runner uses soft_continue)
            info = dict(info or {})
            info["soft_continue"] = True
            return obs, reward, done, info

    return ShapedEnv(env)


def make_option_factory(env: Any, evgs: EVGS, use_ppo: bool, train_options: bool, ppo_steps: int) -> Callable[[Skill], Any]:
    """
    Build a factory callable(skill) -> OptionPolicy.
    If use_ppo and SB3 is available, returns a PPOOption for each skill, optionally pre-trained.
    Otherwise returns RandomOption.
    """
    def factory(skill: Skill):
        if use_ppo and SB3_AVAILABLE:
            opt_cfg = OptionConfig(max_steps=128, terminate_on_success=True, ppo_total_timesteps=ppo_steps, ppo_verbose=0)
            option = PPOOption(skill.subgoal, opt_cfg)
            if train_options:
                shaped = make_shaped_env(env, evgs, skill.subgoal)
                option.train(shaped, reward_wrapper=None, total_timesteps=ppo_steps)
            return option
        # fallback
        return RandomOption(skill.subgoal, OptionConfig(max_steps=128, terminate_on_success=True), env.action_space)
    return factory


# -------------------------- Build a tiny demo SIG --------------------------

def build_demo_sig(evgs: EVGS) -> SIGraph:
    """
    Construct a tiny SIG with two skills as examples:
      - skill 0: increase X0
      - skill 1: decrease X1 (requires skill 0)
    Adjust var indices/predicates for your EVGS.
    """
    sig = SIGraph()
    # Defensive for small obs spaces
    var0 = 0
    var1 = 1 if len(evgs.var_names) > 1 else 0
    sg0 = Subgoal(var_index=var0, predicate=Predicate.UP)
    sg1 = Subgoal(var_index=var1, predicate=Predicate.DOWN)
    sig.add_skill(Skill(skill_id=0, subgoal=sg0, name="increase_X0"))
    sig.add_skill(Skill(skill_id=1, subgoal=sg1, name="decrease_X1"))
    sig.add_prerequisite(0, 1)
    return sig


# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--pcg", type=str, default="notears", choices=["notears", "variational", "simple"])
    parser.add_argument("--steps", type=int, default=200, help="number of DIA macro-steps")
    parser.add_argument("--fit_every", type=int, default=10, help="fit PCG every N macro-steps")
    parser.add_argument("--pcg_epochs", type=int, default=200, help="steps per PCG fit() call")
    parser.add_argument("--buffer_recent", type=int, default=1024)
    parser.add_argument("--min_buffer", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="runs/dia_demo")
    parser.add_argument("--use_ppo_options", action="store_true", help="use PPO options if SB3 installed")
    parser.add_argument("--train_options", action="store_true", help="pre-train each PPO option with shaped reward")
    parser.add_argument("--ppo_steps", type=int, default=5000)
    args = parser.parse_args()

    if gym is None:
        raise RuntimeError("Please install gym/gymnasium to run this script.")

    # Environment
    env = gym.make(args.env_id)
    # EVGS adapter (identity for low-dim)
    if hasattr(env.observation_space, "shape") and env.observation_space.shape is not None:
        obs_dim = int(env.observation_space.shape[0])
    else:
        raise RuntimeError("This demo expects a low-dimensional Box observation space.")
    evgs = build_lowdim_evgs(obs_dim=obs_dim)

    # PCG backend
    M = len(evgs.var_names)
    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=0))
    elif args.pcg == "variational":
        pcg = VariationalPCG(VariationalPCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, K=4, verbose=False))
    else:
        pcg = DifferentiablePCG(DifferentiablePCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, verbose=False))

    # SIG + Planner
    sig = build_demo_sig(evgs)
    selector = InterventionSelector(pcg, sig, PlannerConfig())

    # Logger
    logger = TBLogger(args.logdir)

    # Runner
    rcfg = RunnerConfig(
        buffer_size=10_000,
        min_buffer=args.min_buffer,
        batch_recent=args.buffer_recent,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        log_prefix="dia",
        option_max_steps=64,
        terminate_on_success=True,
    )
    option_factory = make_option_factory(env, evgs, args.use_ppo_options, args.train_options, args.ppo_steps)
    runner = DIARunner(env, evgs, pcg, sig, selector, rcfg, logger=logger, option_factory=option_factory)

    # Loop
    achieved: List[int] = []
    for t in range(args.steps):
        rec = runner.step(achieved, task_goal=None)
        # (Optional) mark a skill achieved if success to unlock prereqs in demo
        if rec["success"]:
            sid = rec["skill_id"]
            if sid not in achieved:
                achieved.append(sid)

    logger.flush()
    logger.close()
    print("Finished. Logs in:", args.logdir)


if __name__ == "__main__":
    main()
