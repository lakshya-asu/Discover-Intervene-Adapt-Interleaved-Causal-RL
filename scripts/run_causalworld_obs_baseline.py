#!/usr/bin/env python3
# scripts/run_causalworld_obs_baseline.py
"""
Observational PCG baseline for CausalWorld.

Approximates CDL/CARL-style purely observational causal discovery by
ignoring the intervention mask: ALL buffer transitions are used to
estimate each edge i->j, regardless of which skill was executed.

This directly tests Paper Claim 1:
  "DIA's interventional PCG update produces more accurate causal graphs
   than purely observational methods."

Expected outcome: observational SHD > DIA interventional SHD, because
observational estimates are confounded by correlated prerequisites
(grasped and target_lifted are co-active in many transitions, creating
false edges between unrelated pairs).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
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
from dia.shaping import PredicateShapingEnv
from dia.eval.pcg_metrics import shd as eval_shd, ece as eval_ece


# ----------------------------- Observational estimator -----------------------------

def _observational_fit_probs(
    X_t: np.ndarray,
    X_tp1: np.ndarray,
    alpha: float = 5.0,
) -> np.ndarray:
    """
    Purely observational causal-edge estimation. Ignores the intervention mask.

    For edge (i -> j): uses ALL buffer transitions (not filtered by which skill
    was executed). Computes:
      P(j↑ at t+1 | i active at t)  vs  P(j↑ at t+1 | i inactive at t)

    This is analogous to CDL (Ke et al., NeurIPS 2021) and CARL (Huang et al.),
    which use observational co-occurrence to estimate causal structure.

    Confounding problem: in a prerequisite chain A->B->C, i=A and j=C are highly
    correlated (A is active whenever B and C are active), so observational P(C↑|A=1)
    inflates, producing false positive A->C edges.

    Same guards as interventional version (min_samples, min_inactive, balance_min)
    are applied for a fair comparison — the only difference is mask filtering.

    Returns a [d, d] matrix: estimated edge probabilities where observed,
    NaN where there is insufficient data (diagonal is 0).
    """
    if X_t.shape[0] == 0:
        d = X_t.shape[1] if X_t.ndim > 1 else 1
        return np.full((d, d), np.nan)

    N, d = X_t.shape
    probs = np.full((d, d), np.nan, dtype=float)
    min_samples = 3
    min_inactive = 15
    balance_min = 0.25

    for j in range(d):
        # Observational: use ALL N transitions (no mask filter)
        j_up = (X_tp1[:, j] > X_t[:, j]).astype(float)

        for i in range(d):
            if i == j:
                continue
            active_i   = (X_t[:, i] > 0.5).astype(float)
            inactive_i = 1.0 - active_i

            n_active   = float(np.sum(active_i))
            n_inactive = float(np.sum(inactive_i))
            n_total    = n_active + n_inactive

            if (n_active < min_samples or n_inactive < min_inactive
                    or n_inactive / n_total < balance_min):
                continue

            # Beta posterior: (successes + alpha) / (trials + 2*alpha)
            p_up_active   = (float(np.sum(active_i   * j_up)) + alpha) / (n_active   + 2 * alpha)
            p_up_inactive = (float(np.sum(inactive_i * j_up)) + alpha) / (n_inactive + 2 * alpha)

            denom = p_up_active + p_up_inactive
            probs[i, j] = p_up_active / denom if denom > 1e-10 else np.nan

    np.fill_diagonal(probs, 0.0)
    return probs


# ----------------------------- Observational DIARunner subclass --------------------

class ObservationalDIARunner(DIARunner):
    """
    DIARunner variant that uses observational (mask-ignoring) PCG estimation.

    Overrides only _maybe_fit_pcg to replace _interventional_fit_probs with
    _observational_fit_probs. All other behavior (buffer, SIG expansion,
    option execution, logging) is identical to the interventional runner.
    """

    def _maybe_fit_pcg(self) -> Tuple[bool, float, float]:
        self.steps += 1
        if self.steps % max(1, self.cfg.fit_every) != 0:
            return False, 0.0, 0.0
        if len(self.buffer) < self.cfg.min_buffer:
            return False, 0.0, 0.0

        packed, _mask = self.buffer.recent(self.cfg.batch_recent)
        if packed.shape[0] == 0:
            return False, 0.0, 0.0

        d = len(self.evgs.var_names)
        X_t_data   = packed[:, :d]
        X_tp1_data = packed[:, d:]

        old_probs   = np.array(getattr(self.pcg, "probs")).copy()
        old_entropy = float(getattr(self.pcg, "entropy")()) if hasattr(self.pcg, "entropy") else np.nan

        # ── Observational causal discovery: ignore mask, use all transitions ───
        # NOTE: _mask is retrieved but deliberately not passed here.
        # This is the key difference from DIARunner._maybe_fit_pcg.
        est_probs = _observational_fit_probs(X_t_data, X_tp1_data)

        # Merge: only update edges where we have real signal
        new_probs = old_probs.copy()
        observed = ~np.isnan(est_probs)
        new_probs[observed] = est_probs[observed]
        np.fill_diagonal(new_probs, 0.0)

        if hasattr(self.pcg, "apply_update"):
            self.pcg.apply_update(new_probs)
        elif hasattr(self.pcg, "fit"):
            try:
                _ = self.pcg.fit(X_tp1_data, mask=None, epochs=self.cfg.pcg_epochs)
                new_probs = np.array(getattr(self.pcg, "probs"))
            except Exception:
                pass

        new_probs   = np.array(getattr(self.pcg, "probs"))
        new_entropy = float(getattr(self.pcg, "entropy")()) if hasattr(self.pcg, "entropy") else np.nan

        ig_update    = self._bernoulli_kl(new_probs, old_probs)
        entropy_drop = float(old_entropy - new_entropy) if (
            not np.isnan(old_entropy) and not np.isnan(new_entropy)
        ) else 0.0

        if self.logger:
            self.logger.add_scalar(f"{self.cfg.log_prefix}/pcg_entropy", new_entropy)
            self.logger.add_scalar(f"{self.cfg.log_prefix}/ig_update", ig_update)
            self.logger.add_scalar(f"{self.cfg.log_prefix}/entropy_drop", entropy_drop)

        # SIG auto-expansion from updated PCG probs (identical to DIARunner)
        if self.cfg.auto_expand_sig and hasattr(self.pcg, "probs"):
            should_expand = (self.cfg.expand_every == 0) or (
                self.steps % max(1, self.cfg.expand_every) == 0
            )
            if should_expand:
                from dia.sig_auto import expand_sig_from_pcg, AutoSIGConfig
                stats = expand_sig_from_pcg(
                    self.sig, self.evgs, self.pcg.probs,
                    AutoSIGConfig(
                        add_threshold=self.cfg.add_threshold,
                        remove_threshold=self.cfg.remove_threshold,
                        create_missing_skills=self.cfg.create_missing_skills,
                        verbose=False,
                    )
                )
                if self.logger:
                    self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_added",   float(stats["added"]))
                    self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_removed",  float(stats["removed"]))
                    self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_created",  float(stats["created_skills"]))

        return True, ig_update, entropy_drop


# ----------------------------- Environment / SIG helpers --------------------------

def build_cw_sig(evgs: EVGS) -> SIGraph:
    names = evgs.names()
    idx = {n: i for i, n in enumerate(names)}
    sig = SIGraph()
    s_g = Skill(skill_id=idx["grasped"],          subgoal=Subgoal(idx["grasped"],          Predicate.UP), name="grasp↑")
    s_l = Skill(skill_id=idx["target_lifted"],    subgoal=Subgoal(idx["target_lifted"],    Predicate.UP), name="lifted↑")
    s_a = Skill(skill_id=idx["target_above_goal"],subgoal=Subgoal(idx["target_above_goal"],Predicate.UP), name="above_goal↑")
    s_s = Skill(skill_id=idx["task_success"],     subgoal=Subgoal(idx["task_success"],     Predicate.UP), name="success↑")
    for s in (s_g, s_l, s_a, s_s):
        sig.add_skill(s)
    sig.add_prerequisite(s_g.skill_id, s_l.skill_id)
    sig.add_prerequisite(s_l.skill_id, s_a.skill_id)
    sig.add_prerequisite(s_a.skill_id, s_s.skill_id)
    return sig


INTERVENTIONS = {
    "T0": {"obstacle": {"size": np.array([0.5, 0.015, 0.02])}},
    "T1": {"obstacle": {"size": np.array([0.5, 0.015, 0.10])}},
    "T2": {"tool_block": {"size": np.array([0.085, 0.085, 0.085])}},
}

GT_EDGES_CW = [
    ("grasped",          "target_lifted"),
    ("target_lifted",    "target_above_goal"),
    ("target_above_goal","task_success"),
]


# ----------------------------- Main ------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Observational PCG baseline for CausalWorld (Claim 1 comparison)"
    )
    ap.add_argument("--steps",        type=int,   default=5000)
    ap.add_argument("--pcg",          type=str,   default="notears",
                    choices=["notears", "variational", "simple"])
    ap.add_argument("--fit_every",    type=int,   default=10)
    ap.add_argument("--pcg_epochs",   type=int,   default=200)
    ap.add_argument("--buffer_recent",type=int,   default=1024)
    ap.add_argument("--min_buffer",   type=int,   default=256)
    ap.add_argument("--logdir",       type=str,   default="runs/causalworld_obs")
    ap.add_argument("--task_goal",    type=str,   default="task_success")
    ap.add_argument("--condition",    type=str,   default=None, choices=["T0", "T1", "T2"],
                    help="Structural intervention condition")
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--out",          type=str,   default=None,
                    help="Path to write JSON result file. "
                         "Defaults to results/logs/obs_{condition}_{seed}.json")
    args = ap.parse_args()

    condition_str = args.condition if args.condition is not None else "none"
    if args.out is None:
        args.out = f"results/logs/obs_{condition_str}_{args.seed}.json"

    if _env_ctor is None:
        raise RuntimeError(
            "CausalWorld not found. Install 'causalworld' and 'pybullet' to run."
        )

    if _task_ctor is not None:
        task = _task_ctor(task_generator_id="pick_and_place")
        env  = _env_ctor(task=task, enable_visualization=False)
    else:
        env = _env_ctor()
    env = wrap_causalworld_env(env, CausalWorldAdapterConfig())

    if args.condition is not None:
        intervention = INTERVENTIONS[args.condition]
        raw_env = env
        while hasattr(raw_env, "env"):
            raw_env = raw_env.env
        if hasattr(raw_env, "do_intervention"):
            raw_env.do_intervention(intervention)
            print(f"[OBS] Applied condition={args.condition} to raw env")
        else:
            try:
                env.do_intervention(intervention)
                print(f"[OBS] Applied intervention condition={args.condition}: {intervention}")
            except Exception as exc:
                print(f"[OBS] Warning: could not apply intervention: {exc}")

    evgs = make_causalworld_evgs()

    M = len(evgs.names())
    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=0))
    elif args.pcg == "variational":
        pcg = VariationalPCG(VariationalPCGConfig(num_vars=M, max_iter=args.pcg_epochs,
                                                   lr=5e-3, K=4, verbose=False))
    else:
        pcg = DifferentiablePCG(DifferentiablePCGConfig(num_vars=M, max_iter=args.pcg_epochs,
                                                         lr=5e-3, verbose=False))

    sig      = build_cw_sig(evgs)
    selector = InterventionSelector(pcg, sig, PlannerConfig())
    logger   = TBLogger(args.logdir)

    rcfg = RunnerConfig(
        buffer_size=50_000,
        min_buffer=args.min_buffer,
        batch_recent=args.buffer_recent,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        log_prefix="obs",
        option_max_steps=200,
        terminate_on_success=True,
        auto_expand_sig=True, add_threshold=0.75, remove_threshold=0.55,
    )

    def obs_to_x(obs):
        return evgs.extract(obs)

    def option_factory(skill: Skill):
        return RandomOption(
            skill.subgoal,
            OptionConfig(max_steps=200, terminate_on_success=True),
            env.action_space,
        )

    # Use ObservationalDIARunner instead of DIARunner
    runner = ObservationalDIARunner(
        env, evgs, pcg, sig, selector, rcfg, logger=logger, option_factory=option_factory
    )

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

    gt_edges_idx = [
        (name_to_idx[src], name_to_idx[dst])
        for src, dst in GT_EDGES_CW
        if src in name_to_idx and dst in name_to_idx
    ]

    final_shd = float(eval_shd(probs, gt_edges_idx))
    final_ece = float(eval_ece(probs, gt_edges_idx))
    task_success_rate = (
        float(task_success_steps / total_option_steps) if total_option_steps > 0 else 0.0
    )
    pcg_entropy_final = float(pcg.entropy()) if hasattr(pcg, "entropy") else float("nan")

    print(f"[OBS] Final SHD={final_shd} ECE={final_ece:.4f} "
          f"task_success_rate={task_success_rate:.4f} H={pcg_entropy_final:.4f}")

    # --- Write JSON output ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "method":              "obs_baseline",
        "condition":           condition_str,
        "seed":                args.seed,
        "steps":               args.steps,
        "final_shd":           final_shd,
        "final_ece":           final_ece,
        "task_success_rate":   task_success_rate,
        "pcg_entropy_final":   pcg_entropy_final,
        "completed":           True,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OBS] Results written to {out_path}")


if __name__ == "__main__":
    main()
