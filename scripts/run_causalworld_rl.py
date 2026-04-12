#!/usr/bin/env python3
# scripts/run_causalworld_rl.py
"""
CausalWorld DIA with actual RL training (PPO options).

Extends train_causalworld_dia.py by replacing RandomOption with
PPOOption trained via stable-baselines3 on shaped reward.  Measures
both SHD (graph quality) AND task_success_rate (task performance).

Key fix: wraps the dict-observation environment in FlatObsEnv before
passing to SB3, which requires a flat numpy observation.

Usage:
    conda run -n dia-minecraft python3 scripts/run_causalworld_rl.py \\
        --seed 0 --steps 20000 --ppo_steps 10000 --condition T0

Output:
    results/logs/cw_rl_{condition}_{seed}.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gym
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
from dia.options import RandomOption, OptionConfig, PPOOption
from dia.shaping import PredicateShapingEnv
from dia.eval.pcg_metrics import shd as eval_shd, ece as eval_ece


# ---------------------------------------------------------------------------
# FlatObsEnv: unwrap nested dict obs for SB3 PPO compatibility
# ---------------------------------------------------------------------------

class FlatObsEnv(gym.Env):
    """
    Wraps the DIA CausalWorld env stack (which returns dict obs)
    into a flat numpy array env compatible with SB3 MlpPolicy.

    The inner CausalWorldInfoWrapper returns {'obs': array, 'info': dict};
    this wrapper extracts obs['obs'] (the raw 67-d CausalWorld state).
    """
    def __init__(self, wrapped_env: Any):
        self._w = wrapped_env
        # Walk to raw env to find the real observation_space
        inner = wrapped_env
        while hasattr(inner, 'env'):
            inner = inner.env
        self.observation_space = inner.observation_space
        self.action_space = wrapped_env.action_space

    def _flatten(self, obs: Any) -> np.ndarray:
        """Extract raw numpy obs from potentially nested dict."""
        if isinstance(obs, dict):
            inner = obs.get('obs', obs)
            if isinstance(inner, dict):
                inner = inner.get('obs', inner)
            return inner
        return obs

    def reset(self) -> np.ndarray:
        obs = self._w.reset()
        return self._flatten(obs)

    def step(self, action: Any):
        obs, r, done, info = self._w.step(action)
        return self._flatten(obs), r, done, info

    def render(self, mode: str = 'human'):
        pass


# ---------------------------------------------------------------------------
# CausalWorld SIG construction
# ---------------------------------------------------------------------------

INTERVENTIONS = {
    "T0": {"obstacle": {"size": np.array([0.5, 0.015, 0.02])}},
    "T1": {"obstacle": {"size": np.array([0.5, 0.015, 0.10])}},
    "T2": {"tool_block": {"size": np.array([0.085, 0.085, 0.085])}},
}

GT_EDGES_CW = [
    ("grasped", "target_lifted"),
    ("target_lifted", "target_above_goal"),
    ("target_above_goal", "task_success"),
]


def build_cw_sig(evgs: EVGS) -> SIGraph:
    """Build SIGraph with 4-skill pick-and-place chain."""
    names = evgs.names()
    idx = {n: i for i, n in enumerate(names)}
    sig = SIGraph()
    s_g = Skill(skill_id=idx["grasped"],           subgoal=Subgoal(idx["grasped"],           Predicate.UP), name="grasp↑")
    s_l = Skill(skill_id=idx["target_lifted"],     subgoal=Subgoal(idx["target_lifted"],     Predicate.UP), name="lifted↑")
    s_a = Skill(skill_id=idx["target_above_goal"], subgoal=Subgoal(idx["target_above_goal"], Predicate.UP), name="above_goal↑")
    s_s = Skill(skill_id=idx["task_success"],      subgoal=Subgoal(idx["task_success"],      Predicate.UP), name="success↑")
    for s in (s_g, s_l, s_a, s_s):
        sig.add_skill(s)
    sig.add_prerequisite(s_g.skill_id, s_l.skill_id)
    sig.add_prerequisite(s_l.skill_id, s_a.skill_id)
    sig.add_prerequisite(s_a.skill_id, s_s.skill_id)
    return sig


# ---------------------------------------------------------------------------
# PPO option factory with FlatObsEnv fix
# ---------------------------------------------------------------------------

def make_ppo_option_factory(
    env: Any,
    evgs: EVGS,
    ppo_steps: int,
    option_max_steps: int = 200,
    n_envs: int = 4,
) -> Any:
    """
    Returns an option_factory function that creates a trained PPOOption
    for each skill.

    Uses FlatObsEnv to fix the dict-obs SB3 incompatibility.
    Uses n_envs parallel environments for faster training.
    Falls back to RandomOption if training fails.

    NOTE: Each option is shaped only for its own subgoal (e.g., grasp only
    trains to lift block slightly above table; it does NOT learn the full
    pick-and-place sequence).  For task_success to be non-zero, the DIA
    loop must sequence these options correctly using the PCG plan.
    """
    trained_models: Dict[int, Any] = {}

    def _make_env_fn(sg: Subgoal):
        """Factory for fresh env instances with the given subgoal shaping."""
        if _task_ctor is not None:
            def fn():
                t = _task_ctor(task_generator_id="pick_and_place")
                r = _env_ctor(task=t, enable_visualization=False)
                e = wrap_causalworld_env(r, CausalWorldAdapterConfig())
                s = PredicateShapingEnv(e, extractor=lambda obs: evgs.extract(obs), subgoal=sg)
                return FlatObsEnv(s)
        else:
            def fn():
                r = _env_ctor()
                e = wrap_causalworld_env(r, CausalWorldAdapterConfig())
                s = PredicateShapingEnv(e, extractor=lambda obs: evgs.extract(obs), subgoal=sg)
                return FlatObsEnv(s)
        return fn

    def option_factory(skill: Skill) -> Any:
        sid = skill.skill_id
        if sid in trained_models:
            return trained_models[sid]

        subgoal = skill.subgoal
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv

            env_fn = _make_env_fn(subgoal)
            vec_env = DummyVecEnv([env_fn] * n_envs)

            model = PPO(
                'MlpPolicy', vec_env,
                n_steps=512, batch_size=128, n_epochs=10,
                learning_rate=3e-4, gamma=0.99,
                verbose=0, device='cpu',
            )
            print(f"  [PPO] Training option for skill={skill.name} ({ppo_steps} steps, {n_envs} envs)...")
            t0 = time.time()
            model.learn(total_timesteps=ppo_steps)
            elapsed = time.time() - t0
            print(f"  [PPO] Done in {elapsed:.1f}s")

            # Create a PPOOption with the trained model.
            # Patch act() to handle dict obs: the DIA runner passes the wrapped
            # dict obs to option.act(), but SB3 expects a flat array.
            opt_cfg = OptionConfig(max_steps=option_max_steps, terminate_on_success=True)
            opt = PPOOption(subgoal, opt_cfg, model=model)

            _model_ref = model

            def act_with_flatten(obs):
                flat_obs = obs
                if isinstance(obs, dict):
                    flat_obs = obs.get('obs', obs)
                    if isinstance(flat_obs, dict):
                        flat_obs = flat_obs.get('obs', flat_obs)
                action, _ = _model_ref.predict(flat_obs, deterministic=True)
                return action

            opt.act = act_with_flatten

            trained_models[sid] = opt
            return opt
        except Exception as exc:
            import traceback
            print(f"  [PPO] WARNING: training failed for skill={skill.name}: {exc}")
            traceback.print_exc()
            print("  [PPO] Falling back to RandomOption")
            fallback = RandomOption(
                skill.subgoal,
                OptionConfig(max_steps=option_max_steps, terminate_on_success=True),
                env.action_space,
            )
            trained_models[sid] = fallback
            return fallback

    return option_factory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CausalWorld DIA with PPO options (RL training)")
    ap.add_argument("--steps",         type=int, default=20000,
                    help="DIA option-level steps (default 20000)")
    ap.add_argument("--ppo_steps",     type=int, default=50000,
                    help="PPO training steps per skill option (default 50000; "
                         "50K achieves reliable grasp in CausalWorld pick_and_place)")
    ap.add_argument("--n_envs",        type=int, default=4,
                    help="Parallel envs for PPO training (default 4)")
    ap.add_argument("--pcg",           type=str, default="notears",
                    choices=["notears", "variational", "simple"])
    ap.add_argument("--fit_every",     type=int, default=10)
    ap.add_argument("--pcg_epochs",    type=int, default=200)
    ap.add_argument("--buffer_recent", type=int, default=1024)
    ap.add_argument("--min_buffer",    type=int, default=256)
    ap.add_argument("--logdir",        type=str, default="runs/causalworld_rl")
    ap.add_argument("--condition",     type=str, default=None, choices=["T0", "T1", "T2"])
    ap.add_argument("--seed",          type=int, default=0)
    ap.add_argument("--out",           type=str, default=None)
    ap.add_argument("--option_max_steps", type=int, default=200)
    args = ap.parse_args()

    np.random.seed(args.seed)

    condition_str = args.condition if args.condition is not None else "none"
    if args.out is None:
        args.out = f"results/logs/cw_rl_{condition_str}_{args.seed}.json"

    if _env_ctor is None:
        raise RuntimeError("CausalWorld not found. Install 'causalworld' and 'pybullet'.")

    print(f"[RL] CausalWorld DIA with PPO options")
    print(f"[RL] condition={condition_str} seed={args.seed} steps={args.steps} ppo_steps={args.ppo_steps}")

    # Build environment
    if _task_ctor is not None:
        task = _task_ctor(task_generator_id="pick_and_place")
        raw_env = _env_ctor(task=task, enable_visualization=False)
    else:
        raw_env = _env_ctor()
    env = wrap_causalworld_env(raw_env, CausalWorldAdapterConfig())

    # Apply structural/motor intervention
    if args.condition is not None:
        intervention = INTERVENTIONS[args.condition]
        inner = env
        while hasattr(inner, 'env'):
            inner = inner.env
        if hasattr(inner, 'do_intervention'):
            inner.do_intervention(intervention)
            print(f"[RL] Applied condition={args.condition}")
        else:
            try:
                env.do_intervention(intervention)
            except Exception as exc:
                print(f"[RL] Warning: could not apply intervention: {exc}")

    evgs = make_causalworld_evgs()
    names = evgs.names()
    print(f"[RL] EVGS variables: {names}")

    # Build PCG
    M = len(names)
    from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
    from dia.pcg_variational import VariationalPCG, VariationalPCGConfig
    from dia.pcg import SimplePCG, PCGConfig

    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=args.seed))
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
        log_prefix="cw_rl",
        option_max_steps=args.option_max_steps,
        terminate_on_success=True,
        auto_expand_sig=True,
        add_threshold=0.75,
        remove_threshold=0.55,
    )

    # PPO option factory: pre-trains one PPOOption per skill
    option_factory = make_ppo_option_factory(
        env=env,
        evgs=evgs,
        ppo_steps=args.ppo_steps,
        option_max_steps=args.option_max_steps,
        n_envs=args.n_envs,
    )

    runner = DIARunner(env, evgs, pcg, sig, selector, rcfg, logger=logger, option_factory=option_factory)

    name_to_idx = {n: i for i, n in enumerate(names)}
    task_goal = Subgoal(var_index=name_to_idx["task_success"], predicate=Predicate.UP)

    achieved: List[int] = []
    task_success_steps = 0
    total_option_steps = 0
    task_completions = 0   # full task_success achieved
    eval_window_steps = 0
    eval_window_successes = 0

    print(f"\n[RL] Starting DIA loop ({args.steps} option steps)...")
    t_start = time.time()

    for t in range(args.steps):
        rec = runner.step(achieved, task_goal=task_goal)
        total_option_steps += 1

        is_task_success = (
            rec.get("success") and
            rec.get("skill_id") == name_to_idx.get("task_success")
        )
        if is_task_success:
            task_success_steps += 1
            task_completions += 1

        # Sliding window for recent performance (last 500 steps)
        eval_window_steps += 1
        if is_task_success:
            eval_window_successes += 1
        if eval_window_steps > 500:
            eval_window_steps -= 1
            if eval_window_successes > 0:
                eval_window_successes -= 1  # approximate FIFO

        if (t + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (t + 1) / elapsed
            remaining = (args.steps - t - 1) / rate if rate > 0 else 0
            print(f"[{t+1:05d}/{args.steps}] phase={rec['phase']} "
                  f"skill={rec['skill_name']:<12} succ={int(rec['success'])} "
                  f"task_compl={task_completions} "
                  f"SHD_live=? H={rec['pcg_entropy']:.3f} "
                  f"buf={rec['buffer_size']} "
                  f"ETA={remaining:.0f}s")

        if rec["success"] and rec["skill_id"] not in achieved:
            achieved.append(rec["skill_id"])

    total_elapsed = time.time() - t_start
    print(f"\n[RL] Finished in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    logger.flush()
    logger.close()

    # --- Compute metrics ---
    probs = pcg.probs if isinstance(pcg.probs, np.ndarray) else pcg.probs()
    name_to_idx = {n: i for i, n in enumerate(names)}
    gt_edges_idx = [
        (name_to_idx[src], name_to_idx[dst])
        for src, dst in GT_EDGES_CW
        if src in name_to_idx and dst in name_to_idx
    ]
    final_shd = float(eval_shd(probs, gt_edges_idx))
    final_ece = float(eval_ece(probs, gt_edges_idx))
    task_success_rate = float(task_success_steps / total_option_steps) if total_option_steps > 0 else 0.0
    pcg_entropy_final = float(pcg.entropy()) if hasattr(pcg, "entropy") else float("nan")

    print(f"[RL] Final: SHD={final_shd} ECE={final_ece:.4f} "
          f"task_success_rate={task_success_rate:.4f} "
          f"task_completions={task_completions} H={pcg_entropy_final:.4f}")

    result = {
        "method": "dia_rl",
        "condition": condition_str,
        "seed": args.seed,
        "steps": args.steps,
        "ppo_steps_per_skill": args.ppo_steps,
        "n_envs": args.n_envs,
        "final_shd": final_shd,
        "final_ece": final_ece,
        "task_success_rate": task_success_rate,
        "task_completions": task_completions,
        "pcg_entropy_final": pcg_entropy_final,
        "total_elapsed_s": total_elapsed,
        "completed": True,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[RL] Results written to {out_path}")


if __name__ == "__main__":
    main()
