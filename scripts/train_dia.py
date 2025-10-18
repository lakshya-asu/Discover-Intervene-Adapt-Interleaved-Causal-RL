#!/usr/bin/env python3
"""
Minimal training script wiring PCG + SIG + Planner + Runner.
This is a skeleton to be adapted to your environments and policies.
"""
import argparse
import time
from typing import List, Optional
import numpy as np

try:
    import gym
except Exception:
    gym = None

from dia.evgs import EVGS  # noqa: E401 (path is resolved when installed or PYTHONPATH set)
from dia.types import Subgoal, Predicate
from dia.pcg import SimplePCG, PCGConfig
from dia.sig import SIGraph, Skill
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner
from dia.logging_utils import TBLogger


def build_dummy_evgs(obs_dim: int = 4):
    # Example adapter: identity observation as variables (CartPole-like low-dim state)
    def obs_to_vars(obs):
        return np.array(obs[:obs_dim], dtype=float)
    return EVGS(var_names=[f"X{i}" for i in range(obs_dim)], obs_to_vars=obs_to_vars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="runs/dia")
    args = parser.parse_args()

    if gym is None:
        raise RuntimeError("Please install gym/gymnasium to run this script.")

    env = gym.make(args.env_id)
    evgs = build_dummy_evgs(obs_dim=4)

    # Build PCG/SIG
    M = len(evgs.var_names)
    pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=0))

    sig = SIGraph()
    # Define a couple of illustrative subgoals on first variable (e.g., increase angle), etc.
    sg0 = Subgoal(var_index=0, predicate=Predicate.UP)
    sg1 = Subgoal(var_index=1, predicate=Predicate.DOWN)
    sig.add_skill(Skill(skill_id=0, subgoal=sg0, name="increase_X0"))
    sig.add_skill(Skill(skill_id=1, subgoal=sg1, name="decrease_X1"))
    # Example prerequisite: 0 -> 1
    sig.add_prerequisite(0, 1)

    selector = InterventionSelector(pcg, sig, PlannerConfig())

    runner = DIARunner(env, evgs, pcg, sig, selector)
    logger = TBLogger(args.logdir)

    achieved: List[int] = []  # track achieved skills by ID (toy)

    for ep in range(args.episodes):
        rec = runner.step(achieved, task_goal=None)
        logger.add_scalar("ig", rec["ig"], ep)
        logger.add_scalar("pcg_entropy", rec["pcg_entropy"], ep)
        logger.add_scalar("skill_success", float(rec["success"]), ep)

    logger.flush()
    logger.close()

    print("Finished. You can inspect logs under:", args.logdir)


if __name__ == "__main__":
    main()
