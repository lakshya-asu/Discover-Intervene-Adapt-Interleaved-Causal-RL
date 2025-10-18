#!/usr/bin/env python3
# scripts/train_causalworld_dia.py
from __future__ import annotations

import argparse
from typing import List
import numpy as np

_env_ctor = None
try:
    from causal_world.envs.causalworld import CausalWorld  # type: ignore
    _env_ctor = CausalWorld
except Exception:
    try:
        from causal_world.envs import CausalWorld  # type: ignore
        _env_ctor = CausalWorld
    except Exception:
        _env_ctor = None

from dia.evgs_causalworld import wrap_causalworld_env, CausalWorldAdapterConfig
from dia.evgs_adapters import make_causalworld_evgs
from dia.evgs import EVGS
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.logging_utils import TBLogger

from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.pcg_variational import VariationalPCG, VariationalPCGConfig
from dia.pcg import SimplePCG, PCGConfig

from dia.options import RandomOption, OptionConfig, PPOOption
from dia.shaping import PredicateShapingEnv  # generic EVGS-like shaping via extractor


def build_cw_sig(evgs: EVGS) -> SIGraph:
    names = evgs.names()
    idx = {n: i for i, n in enumerate(names)}
    sig = SIGraph()
    s_h = Skill(skill_id=idx["tower_height"], subgoal=Subgoal(idx["tower_height"], Predicate.UP), name="height↑")
    s_d = Skill(skill_id=idx["distance_to_goal"], subgoal=Subgoal(idx["distance_to_goal"], Predicate.DOWN), name="goal_dist↓")
    s_g = Skill(skill_id=idx["grasped"], subgoal=Subgoal(idx["grasped"], Predicate.UP), name="grasp↑")
    for s in (s_h, s_d, s_g):
        sig.add_skill(s)
    sig.add_prerequisite(s_g.skill_id, s_h.skill_id)
    sig.add_prerequisite(s_d.skill_id, s_h.skill_id)
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--pcg", type=str, default="notears", choices=["notears", "variational", "simple"])
    ap.add_argument("--fit_every", type=int, default=10)
    ap.add_argument("--pcg_epochs", type=int, default=200)
    ap.add_argument("--buffer_recent", type=int, default=1024)
    ap.add_argument("--min_buffer", type=int, default=256)
    ap.add_argument("--logdir", type=str, default="runs/causalworld_dia")
    ap.add_argument("--task_goal", type=str, default="tower_height")  # goal-aware planner
    ap.add_argument("--use_ppo_options", action="store_true")
    ap.add_argument("--train_options", action="store_true")
    ap.add_argument("--ppo_steps", type=int, default=5000)
    args = ap.parse_args()

    if _env_ctor is None:
        raise RuntimeError("CausalWorld not found. Install 'causalworld' and 'pybullet' to run this demo.")

    env = _env_ctor()
    env = wrap_causalworld_env(env, CausalWorldAdapterConfig())
    evgs = make_causalworld_evgs()

    M = len(evgs.names())
    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=0))
    elif args.pcg == "variational":
        pcg = VariationalPCG(VariationalPCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, K=4, verbose=False))
    else:
        pcg = DifferentiablePCG(DifferentiablePCGConfig(num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, verbose=False))

    sig = build_cw_sig(evgs)
    selector = InterventionSelector(pcg, sig, PlannerConfig())

    logger = TBLogger(args.logdir)
    rcfg = RunnerConfig(
        buffer_size=50_000,
        min_buffer=args.min_buffer,
        batch_recent=args.buffer_recent,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        log_prefix="cw",
        option_max_steps=64,
        terminate_on_success=True,
        auto_expand_sig=True, add_threshold=0.75, remove_threshold=0.55,
    )

    # PPO options with generic EVGS-style shaping: we need an extractor from obs->X
    def obs_to_x(obs):
        # For wrapped env, obs is {"obs": base_obs, "info": {...}}, and our EVGS reads from info
        return evgs.extract(obs)

    def option_factory(skill: Skill):
        if args.use_ppo_options:
            try:
                shaped = PredicateShapingEnv(env, extractor=obs_to_x, subgoal=skill.subgoal)
                opt_cfg = OptionConfig(max_steps=64, terminate_on_success=True, ppo_total_timesteps=args.ppo_steps)
                opt = PPOOption(skill.subgoal, opt_cfg)
                if args.train_options:
                    opt.train(shaped, reward_wrapper=None, total_timesteps=args.ppo_steps)
                return opt
            except Exception:
                pass
        return RandomOption(skill.subgoal, OptionConfig(max_steps=64, terminate_on_success=True), env.action_space)

    runner = DIARunner(env, evgs, pcg, sig, selector, rcfg, logger=logger, option_factory=option_factory)

    name_to_idx = {n: i for i, n in enumerate(evgs.names())}
    goal_name = args.task_goal if args.task_goal in name_to_idx else "tower_height"
    pred = Predicate.DOWN if goal_name == "distance_to_goal" else Predicate.UP
    task_goal = Subgoal(var_index=name_to_idx[goal_name], predicate=pred)

    achieved: List[int] = []
    for t in range(args.steps):
        rec = runner.step(achieved, task_goal=task_goal)
        if (t + 1) % 10 == 0:
            print(f"[{t+1:04d}] phase={rec['phase']} skill={rec['skill_name']:<12} "
                  f"succ={int(rec['success'])} IGfit={rec['ig_update']:.4f} "
                  f"H={rec['pcg_entropy']:.4f} buf={rec['buffer_size']}")
        if rec["success"] and rec["skill_id"] not in achieved:
            achieved.append(rec["skill_id"])

    logger.flush()
    logger.close()
    print("Finished. Logs in:", args.logdir)


if __name__ == "__main__":
    main()
