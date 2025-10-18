#!/usr/bin/env python3
# scripts/train_coinrun_dia.py
from __future__ import annotations

import argparse
from typing import List
import numpy as np

try:
    import gym
except Exception:
    gym = None

from dia.evgs_procgen import wrap_procgen_coinrun_env, CoinRunDetectorConfig
from dia.evgs_adapters import make_coinrun_evgs
from dia.evgs import EVGS
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.logging_utils import TBLogger

from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.pcg_variational import VariationalPCG, VariationalPCGConfig
from dia.pcg import SimplePCG, PCGConfig

from dia.options import RandomOption, OptionConfig
from dia.options import PPOOption
from dia.shaping import CoinCollectedShaping  # simple coin shaping (info-based)


def build_coinrun_sig(evgs: EVGS) -> SIGraph:
    names = evgs.names()
    idx = {n: i for i, n in enumerate(names)}
    sig = SIGraph()
    s_coin = Skill(skill_id=idx.get("coin_collected", 0), subgoal=Subgoal(idx.get("coin_collected", 0), Predicate.UP), name="coin↑")
    s_prog = Skill(skill_id=idx.get("progress", 1), subgoal=Subgoal(idx.get("progress", 1), Predicate.UP), name="progress↑")
    s_enemy = Skill(skill_id=idx.get("enemy_near", 2), subgoal=Subgoal(idx.get("enemy_near", 2), Predicate.DOWN), name="enemy_far")
    for s in (s_coin, s_prog, s_enemy):
        if s.skill_id not in sig.skills:
            sig.add_skill(s)
    sig.add_prerequisite(s_prog.skill_id, s_coin.skill_id)
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--env_id", type=str, default="procgen:procgen-coinrun-v0")
    ap.add_argument("--pcg", type=str, default="notears", choices=["notears", "variational", "simple"])
    ap.add_argument("--fit_every", type=int, default=10)
    ap.add_argument("--pcg_epochs", type=int, default=200)
    ap.add_argument("--buffer_recent", type=int, default=1024)
    ap.add_argument("--min_buffer", type=int, default=256)
    ap.add_argument("--logdir", type=str, default="runs/coinrun_dia")
    ap.add_argument("--task_goal", type=str, default="coin_collected")
    ap.add_argument("--use_ppo_options", action="store_true")
    ap.add_argument("--train_options", action="store_true")
    ap.add_argument("--ppo_steps", type=int, default=5000)
    args = ap.parse_args()

    if gym is None:
        raise RuntimeError("Please install gym/gymnasium and procgen (pip install procgen).")

    env = gym.make(args.env_id, render_mode="rgb_array")
    env = wrap_procgen_coinrun_env(env, CoinRunDetectorConfig(horizon=512))
    evgs = make_coinrun_evgs()

    M = len(evgs.names())
    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=0))
    elif args.pcg == "variational":
        pcg = VariationalPCG(VariationalPCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, K=4, verbose=False))
    else:
        pcg = DifferentiablePCG(DifferentiablePCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, verbose=False))

    sig = build_coinrun_sig(evgs)
    selector = InterventionSelector(pcg, sig, PlannerConfig())

    logger = TBLogger(args.logdir)
    rcfg = RunnerConfig(
        buffer_size=20_000,
        min_buffer=args.min_buffer,
        batch_recent=args.buffer_recent,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        log_prefix="coinrun",
        option_max_steps=32,
        terminate_on_success=True,
        auto_expand_sig=True, add_threshold=0.75, remove_threshold=0.55,
    )

    def option_factory(skill: Skill):
        if args.use_ppo_options and skill.name.startswith("coin"):
            try:
                # For PPO training, use a simple info-based shaping: +1 when level complete
                shaped = CoinCollectedShaping(env)
                opt_cfg = OptionConfig(max_steps=32, terminate_on_success=True, ppo_total_timesteps=args.ppo_steps)
                opt = PPOOption(skill.subgoal, opt_cfg)
                if args.train_options:
                    opt.train(shaped, reward_wrapper=None, total_timesteps=args.ppo_steps)
                return opt
            except Exception:
                pass
        return RandomOption(skill.subgoal, OptionConfig(max_steps=32, terminate_on_success=True), env.action_space)

    runner = DIARunner(env, evgs, pcg, sig, selector, rcfg, logger=logger, option_factory=option_factory)

    name_to_idx = {n: i for i, n in enumerate(evgs.names())}
    goal_name = args.task_goal if args.task_goal in name_to_idx else "coin_collected"
    task_goal = Subgoal(var_index=name_to_idx[goal_name], predicate=Predicate.UP)

    achieved: List[int] = []
    for t in range(args.steps):
        rec = runner.step(achieved, task_goal=task_goal)
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
