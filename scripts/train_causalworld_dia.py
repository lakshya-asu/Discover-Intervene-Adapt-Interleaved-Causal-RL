#!/usr/bin/env python3
# scripts/train_causalworld_dia.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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

_task_ctor = None
try:
    from causal_world.task_generators import generate_task  # type: ignore
    _task_ctor = generate_task
except Exception:
    _task_ctor = None

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
from dia.eval.pcg_metrics import shd as eval_shd, ece as eval_ece


def build_cw_sig(evgs: EVGS) -> SIGraph:
    names = evgs.names()
    idx = {n: i for i, n in enumerate(names)}
    sig = SIGraph()
    # Skills matching the 5-variable pick_and_place EVGS
    s_g = Skill(skill_id=idx["grasped"], subgoal=Subgoal(idx["grasped"], Predicate.UP), name="grasp↑")
    s_l = Skill(skill_id=idx["target_lifted"], subgoal=Subgoal(idx["target_lifted"], Predicate.UP), name="lifted↑")
    s_a = Skill(skill_id=idx["target_above_goal"], subgoal=Subgoal(idx["target_above_goal"], Predicate.UP), name="above_goal↑")
    s_s = Skill(skill_id=idx["task_success"], subgoal=Subgoal(idx["task_success"], Predicate.UP), name="success↑")
    for s in (s_g, s_l, s_a, s_s):
        sig.add_skill(s)
    # Prior causal chain: grasped -> lifted -> above_goal -> success
    sig.add_prerequisite(s_g.skill_id, s_l.skill_id)
    sig.add_prerequisite(s_l.skill_id, s_a.skill_id)
    sig.add_prerequisite(s_a.skill_id, s_s.skill_id)
    return sig


INTERVENTIONS = {
    "T0": {"obstacle": {"size": np.array([0.5, 0.015, 0.02])}},
    "T1": {"obstacle": {"size": np.array([0.5, 0.015, 0.10])}},
    "T2": {"tool_block": {"size": np.array([0.085, 0.085, 0.085])}},
}

# Ground-truth causal edges for pick_and_place (4-skill chain)
GT_EDGES_CW = [
    ("grasped", "target_lifted"),
    ("target_lifted", "target_above_goal"),
    ("target_above_goal", "task_success"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--pcg", type=str, default="notears", choices=["notears", "variational", "simple"])
    ap.add_argument("--fit_every", type=int, default=10)
    ap.add_argument("--pcg_epochs", type=int, default=200)
    ap.add_argument("--buffer_recent", type=int, default=1024)
    ap.add_argument("--min_buffer", type=int, default=256)
    ap.add_argument("--logdir", type=str, default="runs/causalworld_dia")
    ap.add_argument("--task_goal", type=str, default="task_success")  # goal-aware planner
    ap.add_argument("--use_ppo_options", action="store_true")
    ap.add_argument("--train_options", action="store_true")
    ap.add_argument("--ppo_steps", type=int, default=5000)
    ap.add_argument("--condition", type=str, default=None, choices=["T0", "T1", "T2"],
                    help="Structural intervention condition: T0 (small obstacle), "
                         "T1 (large obstacle, structural), T2 (large tool_block, motor)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility (use 0-4 for 5-seed runs)")
    ap.add_argument("--out", type=str, default=None,
                    help="Path to write JSON result file. "
                         "Defaults to results/logs/cw_{condition}_{seed}.json")
    args = ap.parse_args()

    # Resolve default output path
    condition_str = args.condition if args.condition is not None else "none"
    if args.out is None:
        args.out = f"results/logs/cw_{condition_str}_{args.seed}.json"

    if _env_ctor is None:
        raise RuntimeError("CausalWorld not found. Install 'causalworld' and 'pybullet' to run this demo.")

    if _task_ctor is not None:
        task = _task_ctor(task_generator_id="pick_and_place")
        env = _env_ctor(task=task, enable_visualization=False)
    else:
        env = _env_ctor()
    env = wrap_causalworld_env(env, CausalWorldAdapterConfig())

    # Apply structural/motor intervention to the underlying CausalWorld env,
    # walking through all wrappers to reach the raw env.
    if args.condition is not None:
        intervention = INTERVENTIONS[args.condition]
        raw_env = env
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env
        if hasattr(raw_env, 'do_intervention'):
            raw_env.do_intervention(intervention)
            print(f"[DIA] Applied condition={args.condition} to raw env")
        else:
            # Fallback: try on the top-level wrapper
            try:
                env.do_intervention(intervention)
                print(f"[DIA] Applied intervention condition={args.condition}: {intervention}")
            except Exception as exc:
                print(f"[DIA] Warning: could not apply intervention: {exc}")

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
    # Use option_max_steps=200 for CausalWorld: continuous control needs more steps
    # for random walk to occasionally achieve task-relevant sub-goals.
    rcfg = RunnerConfig(
        buffer_size=50_000,
        min_buffer=args.min_buffer,
        batch_recent=args.buffer_recent,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        log_prefix="cw",
        option_max_steps=200,
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
                opt_cfg = OptionConfig(max_steps=200, terminate_on_success=True, ppo_total_timesteps=args.ppo_steps)
                opt = PPOOption(skill.subgoal, opt_cfg)
                if args.train_options:
                    opt.train(shaped, reward_wrapper=None, total_timesteps=args.ppo_steps)
                return opt
            except Exception:
                pass
        # Dense random walk with 200 max_steps gives CausalWorld's continuous
        # action space a reasonable chance of accidental fractional success.
        return RandomOption(skill.subgoal, OptionConfig(max_steps=200, terminate_on_success=True), env.action_space)

    runner = DIARunner(env, evgs, pcg, sig, selector, rcfg, logger=logger, option_factory=option_factory)

    name_to_idx = {n: i for i, n in enumerate(evgs.names())}
    goal_name = args.task_goal if args.task_goal in name_to_idx else "task_success"
    pred = Predicate.DOWN if goal_name == "target_above_goal" else Predicate.UP
    task_goal = Subgoal(var_index=name_to_idx[goal_name], predicate=pred)

    achieved: List[int] = []
    task_success_steps = 0
    total_option_steps = 0

    for t in range(args.steps):
        rec = runner.step(achieved, task_goal=task_goal)
        total_option_steps += 1
        # Track task_success increases as a proxy for option-level task success
        if rec.get("success") and rec.get("skill_id") == name_to_idx.get("task_success"):
            task_success_steps += 1
        if (t + 1) % 10 == 0:
            print(f"[{t+1:04d}] phase={rec['phase']} skill={rec['skill_name']:<12} "
                  f"succ={int(rec['success'])} IGfit={rec['ig_update']:.4f} "
                  f"H={rec['pcg_entropy']:.4f} buf={rec['buffer_size']}")
        if rec["success"] and rec["skill_id"] not in achieved:
            achieved.append(rec["skill_id"])

    logger.flush()
    logger.close()
    print("Finished. Logs in:", args.logdir)

    # --- Compute final metrics ---
    probs = pcg.probs if isinstance(pcg.probs, np.ndarray) else pcg.probs()

    # Convert GT_EDGES_CW name tuples to index tuples
    gt_edges_idx = [
        (name_to_idx[src], name_to_idx[dst])
        for src, dst in GT_EDGES_CW
        if src in name_to_idx and dst in name_to_idx
    ]

    final_shd = float(eval_shd(probs, gt_edges_idx))
    final_ece = float(eval_ece(probs, gt_edges_idx))
    task_success_rate = float(task_success_steps / total_option_steps) if total_option_steps > 0 else 0.0
    pcg_entropy_final = float(pcg.entropy()) if hasattr(pcg, "entropy") else float("nan")

    print(f"[DIA] Final SHD={final_shd} ECE={final_ece:.4f} "
          f"task_success_rate={task_success_rate:.4f} H={pcg_entropy_final:.4f}")

    # --- Write JSON output ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "dia",
        "condition": condition_str,
        "seed": args.seed,
        "steps": args.steps,
        "final_shd": final_shd,
        "final_ece": final_ece,
        "task_success_rate": task_success_rate,
        "pcg_entropy_final": pcg_entropy_final,
        "completed": True,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[DIA] Results written to {out_path}")


if __name__ == "__main__":
    main()
