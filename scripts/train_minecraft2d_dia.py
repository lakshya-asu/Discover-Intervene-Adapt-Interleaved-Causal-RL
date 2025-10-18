#!/usr/bin/env python3
# scripts/train_minecraft2d_dia.py
from __future__ import annotations

import argparse
from typing import List

from dia.envs.minecraft2d import MinecraftChainEnv, ChainConfig
from dia.evgs_minecraft import make_minecraft_evgs, VAR_NAMES
from dia.evgs import EVGS
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.logging_utils import TBLogger

from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.pcg_variational import VariationalPCG, VariationalPCGConfig
from dia.pcg import SimplePCG, PCGConfig

from dia.options import OptionConfig
from dia.options_minecraft import ResourceOption  # resource-aware micro-policy


def build_minecraft_sig(evgs: EVGS) -> SIGraph:
    idx = {name: i for i, name in enumerate(evgs.names())}
    sig = SIGraph()

    def add_skill(name: str):
        sid = idx[name]
        sg = Subgoal(var_index=sid, predicate=Predicate.UP)
        sig.add_skill(Skill(skill_id=sid, subgoal=sg, name=f"{name}↑"))

    for name in VAR_NAMES:
        add_skill(name)

    # prereqs
    sig.add_prerequisite(idx["stone"], idx["furnace"])
    sig.add_prerequisite(idx["wood"], idx["stonepickaxe"])
    sig.add_prerequisite(idx["stone"], idx["stonepickaxe"])
    sig.add_prerequisite(idx["stonepickaxe"], idx["ironore"])
    sig.add_prerequisite(idx["furnace"], idx["iron"])
    sig.add_prerequisite(idx["coal"], idx["iron"])
    sig.add_prerequisite(idx["ironore"], idx["iron"])
    sig.add_prerequisite(idx["iron"], idx["ironpickaxe"])
    sig.add_prerequisite(idx["wood"], idx["ironpickaxe"])
    sig.add_prerequisite(idx["ironpickaxe"], idx["diamond"])
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--pcg", type=str, default="notears", choices=["notears", "variational", "simple"])
    ap.add_argument("--fit_every", type=int, default=10)
    ap.add_argument("--pcg_epochs", type=int, default=200)
    ap.add_argument("--buffer_recent", type=int, default=1024)
    ap.add_argument("--min_buffer", type=int, default=256)
    ap.add_argument("--logdir", type=str, default="runs/mc2d_dia")
    ap.add_argument("--task_goal", type=str, default="diamond")  # goal-aware selection

    # These flags are accepted for CLI compatibility but are unused in this script
    ap.add_argument("--use_ppo_options", action="store_true")
    ap.add_argument("--train_options", action="store_true")
    ap.add_argument("--ppo_steps", type=int, default=3000)

    args = ap.parse_args()

    env = MinecraftChainEnv(ChainConfig())
    evgs = make_minecraft_evgs()

    M = len(evgs.names())
    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=0))
    elif args.pcg == "variational":
        pcg = VariationalPCG(VariationalPCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, K=4, verbose=False))
    else:
        pcg = DifferentiablePCG(DifferentiablePCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, verbose=False))

    sig = build_minecraft_sig(evgs)
    selector = InterventionSelector(pcg, sig, PlannerConfig())

    logger = TBLogger(args.logdir)
    rcfg = RunnerConfig(
        buffer_size=20_000,
        min_buffer=args.min_buffer,
        batch_recent=args.buffer_recent,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        log_prefix="mc2d",
        option_max_steps=32,
        terminate_on_success=True,
        auto_expand_sig=True, add_threshold=0.75, remove_threshold=0.55,
    )

    # Option factory: ResourceOption (heuristic micro-policy)
    name_to_idx = {n: i for i, n in enumerate(evgs.names())}

    def option_factory(skill: Skill):
        # Default: resource-aware heuristic
        return ResourceOption(skill.subgoal, OptionConfig(max_steps=32, terminate_on_success=True))

    runner = DIARunner(env, evgs, pcg, sig, selector, rcfg, logger=logger, option_factory=option_factory)

    # Goal
    goal_var = args.task_goal if args.task_goal in name_to_idx else "diamond"
    task_goal = Subgoal(var_index=name_to_idx[goal_var], predicate=Predicate.UP)

    achieved: List[int] = []
    for t in range(args.steps):
        rec = runner.step(achieved, task_goal=task_goal)
        if (t + 1) % 10 == 0:
            print(
                f"[{t+1:04d}] phase={rec['phase']} skill={rec['skill_name']:<12} "
                f"succ={int(rec['success'])} IGfit={rec['ig_update']:.4f} "
                f"H={rec['pcg_entropy']:.4f} buf={rec['buffer_size']}"
            )
        if rec["success"] and rec["skill_id"] not in achieved:
            achieved.append(rec["skill_id"])

    logger.flush()
    logger.close()
    print("Finished. Logs in:", args.logdir)


if __name__ == "__main__":
    main()
