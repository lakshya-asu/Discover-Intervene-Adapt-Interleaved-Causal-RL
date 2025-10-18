#!/usr/bin/env python3
# scripts/train_cartpole_dia.py
from __future__ import annotations

import argparse
from typing import List
import numpy as np

try:
    import gym
except Exception:
    gym = None

from dia.evgs_cartpole import make_cartpole_evgs
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.logging_utils import TBLogger

from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.pcg_variational import VariationalPCG, VariationalPCGConfig
from dia.pcg import SimplePCG, PCGConfig
from dia.options import RandomOption, OptionConfig


def build_cp_sig(evgs) -> SIGraph:
    # Simple diagnostic skills: push x to the right (x UP), reduce |theta| (theta REACH 0), slow theta_dot (DOWN magnitude)
    names = evgs.names()
    idx = {n: i for i, n in enumerate(names)}
    sig = SIGraph()
    sig.add_skill(Skill(skill_id=0, subgoal=Subgoal(idx["x"], Predicate.UP), name="x_right"))
    sig.add_skill(Skill(skill_id=2, subgoal=Subgoal(idx["theta"], Predicate.EQUAL, value=0.0), name="theta_zero"))
    sig.add_skill(Skill(skill_id=3, subgoal=Subgoal(idx["theta"], Predicate.DOWN), name="theta_down"))  # coarse
    # trivial prerequisites (none)
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="CartPole-v1")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--pcg", type=str, default="notears", choices=["notears", "variational", "simple"])
    ap.add_argument("--fit_every", type=int, default=10)
    ap.add_argument("--pcg_epochs", type=int, default=200)
    ap.add_argument("--buffer_recent", type=int, default=512)
    ap.add_argument("--min_buffer", type=int, default=64)
    ap.add_argument("--logdir", type=str, default="runs/cartpole_dia")
    args = ap.parse_args()

    if gym is None:
        raise RuntimeError("Please install gym/gymnasium to run this script.")

    env = gym.make(args.env_id)
    evgs = make_cartpole_evgs()

    M = len(evgs.names())
    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=0))
    elif args.pcg == "variational":
        pcg = VariationalPCG(VariationalPCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, K=4, verbose=False))
    else:
        pcg = DifferentiablePCG(DifferentiablePCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, verbose=False))

    sig = build_cp_sig(evgs)
    selector = InterventionSelector(pcg, sig, PlannerConfig())

    logger = TBLogger(args.logdir)
    rcfg = RunnerConfig(
        buffer_size=10_000,
        min_buffer=args.min_buffer,
        batch_recent=args.buffer_recent,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        log_prefix="cp",
        option_max_steps=32,
        terminate_on_success=True,
    )

    def option_factory(skill: Skill):
        return RandomOption(skill.subgoal, OptionConfig(max_steps=32, terminate_on_success=True), env.action_space)

    runner = DIARunner(env, evgs, pcg, sig, selector, rcfg, logger=logger, option_factory=option_factory)

    achieved: List[int] = []
    for t in range(args.steps):
        rec = runner.step(achieved, task_goal=None)
        if (t + 1) % 10 == 0:
            print(f"[{t+1:04d}] phase={rec['phase']} skill={rec['skill_name']:<10} "
                  f"succ={int(rec['success'])} IGfit={rec['ig_update']:.4f} "
                  f"H={rec['pcg_entropy']:.4f} buf={rec['buffer_size']}")
        if rec["success"] and rec["skill_id"] not in achieved:
            achieved.append(rec["skill_id"])

    logger.flush()
    logger.close()
    print("Finished. Logs in:", args.logdir)


if __name__ == "__main__":
    main()
