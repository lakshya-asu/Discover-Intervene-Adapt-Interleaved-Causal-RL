#!/usr/bin/env python3
# scripts/train_montezuma_dia.py
from __future__ import annotations

import argparse
from typing import List
import numpy as np

# Prefer Gym (our wrappers are Gym-style), but we can fall back to Gymnasium if necessary.
def _make_env(env_id_arg: str | None):
    # Try Gym + ALE registration
    try:
        import gym  # Gym 0.26.x
        try:
            import ale_py.gym  # registers ALE namespace with Gym
        except Exception:
            pass
        preferred = env_id_arg or "ALE/MontezumaRevenge-v5"
        try:
            env = gym.make(preferred, render_mode="rgb_array")
            return env, preferred, "gym"
        except Exception as e1:
            # Try older atari-py id (only if atari-py is installed)
            try:
                import atari_py  # noqa: F401
                env = gym.make("MontezumaRevengeNoFrameskip-v4")
                return env, "MontezumaRevengeNoFrameskip-v4", "gym"
            except Exception:
                # Fall through to Gymnasium
                raise e1
    except Exception:
        pass

    # Fallback: Gymnasium + ALE registration
    try:
        import gymnasium as gym  # noqa: F401
        try:
            import ale_py.gymnasium  # registers ALE namespace with Gymnasium
        except Exception:
            pass
        preferred = env_id_arg or "ALE/MontezumaRevenge-v5"
        env = gym.make(preferred, render_mode="rgb_array")
        return env, preferred, "gymnasium"
    except Exception as e2:
        raise RuntimeError(
            "Failed to create Montezuma env. Try:\n"
            "  pip install gymnasium ale-py autorom\n"
            "  python -m AutoROM --accept-license\n"
            "Or use the legacy env: conda env create -f environment-sb3-legacy.yml"
        ) from e2


from dia.evgs_adapters import make_montezuma_evgs
from dia.evgs_montezuma import wrap_montezuma_env, MontezumaDetectorsConfig

from dia.evgs import EVGS
from dia.types import Subgoal, Predicate
from dia.sig import SIGraph, Skill
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.logging_utils import TBLogger

from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.pcg_variational import VariationalPCG, VariationalPCGConfig
from dia.pcg import SimplePCG, PCGConfig

from dia.options import OptionConfig, RandomOption


def build_montezuma_sig(evgs: EVGS) -> SIGraph:
    idx_has_key = 0
    idx_door = 1 if len(evgs.var_names) > 1 else 0
    sig = SIGraph()
    sig.add_skill(Skill(skill_id=0, subgoal=Subgoal(idx_has_key, Predicate.UP), name="get_key"))
    sig.add_skill(Skill(skill_id=1, subgoal=Subgoal(idx_door, Predicate.UP), name="open_door"))
    sig.add_prerequisite(0, 1)
    return sig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default=None)
    parser.add_argument("--pcg", type=str, default="notears", choices=["notears", "variational", "simple"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fit_every", type=int, default=10)
    parser.add_argument("--pcg_epochs", type=int, default=200)
    parser.add_argument("--buffer_recent", type=int, default=1024)
    parser.add_argument("--min_buffer", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="runs/monte_dia")
    parser.add_argument("--task_goal", type=str, default="has_key")
    parser.add_argument("--ppo_steps", type=int, default=3000)  # accepted but unused
    args = parser.parse_args()

    env, used_id, backend = _make_env(args.env_id)
    env = wrap_montezuma_env(env, MontezumaDetectorsConfig())
    evgs = make_montezuma_evgs()

    M = len(evgs.var_names)
    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=0))
    elif args.pcg == "variational":
        pcg = VariationalPCG(VariationalPCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, K=4, verbose=False))
    else:
        pcg = DifferentiablePCG(DifferentiablePCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, verbose=False))

    sig = build_montezuma_sig(evgs)
    selector = InterventionSelector(pcg, sig, PlannerConfig())
    logger = TBLogger(args.logdir)

    rcfg = RunnerConfig(
        buffer_size=20_000,
        min_buffer=args.min_buffer,
        batch_recent=args.buffer_recent,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        log_prefix="monte",
        option_max_steps=64,
        terminate_on_success=True,
        auto_expand_sig=True, add_threshold=0.75, remove_threshold=0.55,
    )

    def option_factory(skill: Skill):
        return RandomOption(skill.subgoal, OptionConfig(max_steps=64, terminate_on_success=True), env.action_space)

    runner = DIARunner(env, evgs, pcg, sig, selector, rcfg, logger=logger, option_factory=option_factory)

    name_to_idx = {n: i for i, n in enumerate(evgs.names())}
    goal_name = args.task_goal if args.task_goal in name_to_idx else "has_key"
    task_goal = Subgoal(var_index=name_to_idx[goal_name], predicate=Predicate.UP)

    achieved: List[int] = []
    for t in range(args.steps):
        rec = runner.step(achieved, task_goal=task_goal)
        if (t + 1) % 10 == 0:
            print(f"[{t+1:04d}] backend={backend} id={used_id} phase={rec['phase']} skill={rec['skill_name']:<10} "
                  f"succ={int(rec['success'])} IGfit={rec['ig_update']:.4f} "
                  f"H={rec['pcg_entropy']:.4f} buf={rec['buffer_size']}")
        if rec["success"] and rec["skill_id"] not in achieved:
            achieved.append(rec["skill_id"])

    logger.flush()
    logger.close()
    print("Finished. Logs in:", args.logdir)


if __name__ == "__main__":
    main()
