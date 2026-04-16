#!/usr/bin/env python3
"""Iterative eval-and-tune driver for DIA+GROOT.

Reads the latest pilot result JSON, runs the eval test suite,
prints a diagnostic, then emits the command for the next tuned run.

Usage:
    python scripts/eval_and_tune.py                       # auto-find latest result
    python scripts/eval_and_tune.py --result /tmp/foo.json
    python scripts/eval_and_tune.py --launch              # also launch next iteration
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT   = Path(__file__).resolve().parent.parent
EVAL_DIR    = REPO_ROOT / "data" / "eval"
TUNE_LOG    = EVAL_DIR / "tune_history.jsonl"

# Import tier constants from the test module
sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_pilot_eval import (
    TIER_BASELINE, TIER_ACCEPTABLE, TIER_TARGET, TIER_STRETCH,
    GATHER_SKILLS, CRAFT_SKILLS, SKILL_ORDER,
    MAX_STEPS_STUCK_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Tuning rules — ordered from most conservative to most aggressive
# ---------------------------------------------------------------------------

# Default pilot params
DEFAULTS = dict(
    mode="dia",
    max_steps_per_skill=2000,
    max_total_steps=40000,
)

# Tuning adjustments keyed by (tier_reached, diagnosis)
TUNING_RULES = [
    # --- fully stuck: all gather skills hit cap --------------------------------
    {
        "name": "increase_steps_aggressively",
        "condition": lambda r, d: d["frac_stuck"] >= 1.0,
        "params": {"max_steps_per_skill": 3000, "max_total_steps": 60000},
        "reason": "Every gather skill hit the step cap — agent may need more time to locate resources",
    },
    # --- below baseline: try more steps + dia_online --------------------------
    {
        "name": "below_baseline_dia_online",
        "condition": lambda r, d: r["n_achieved"] < TIER_BASELINE and d["frac_stuck"] < 1.0,
        "params": {"mode": "dia_online", "max_steps_per_skill": 2500},
        "reason": "0 skills achieved with non-stuck behavior — enable online PCG update",
    },
    # --- baseline but not acceptable: more steps for hard gather skills -------
    {
        "name": "more_steps_hard_gather",
        "condition": lambda r, d: TIER_BASELINE <= r["n_achieved"] < TIER_ACCEPTABLE,
        "params": {"max_steps_per_skill": 2500, "max_total_steps": 50000},
        "reason": f"n_achieved {TIER_BASELINE}–{TIER_ACCEPTABLE-1}: increase steps for harder gathers",
    },
    # --- acceptable but not target: switch to dia_online ----------------------
    {
        "name": "dia_online_for_target",
        "condition": lambda r, d: TIER_ACCEPTABLE <= r["n_achieved"] < TIER_TARGET,
        "params": {"mode": "dia_online", "max_steps_per_skill": 2000},
        "reason": "Acceptable tier reached — enable online PCG to improve skill ordering",
    },
    # --- target: push for stretch with more steps ----------------------------
    {
        "name": "stretch_push",
        "condition": lambda r, d: r["n_achieved"] >= TIER_TARGET and not r["diamond_reached"],
        "params": {"mode": "dia_online", "max_steps_per_skill": 3000, "max_total_steps": 60000},
        "reason": "Target tier reached — push for diamond with extra steps",
    },
]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose(result: Dict[str, Any]) -> Dict[str, Any]:
    """Compute diagnostic metrics from a result dict."""
    gather_recs = [
        rec for sk, rec in result["results"].items()
        if sk in GATHER_SKILLS and rec.get("steps", 0) > 0
    ]
    stuck = [r for r in gather_recs if r["steps"] >= MAX_STEPS_STUCK_THRESHOLD]
    frac_stuck = len(stuck) / max(len(gather_recs), 1)

    # Per-skill success breakdown
    skill_successes = {sk: result["results"].get(sk, {}).get("success", False)
                       for sk in SKILL_ORDER}

    # First failed skill in order
    first_fail = next((sk for sk in SKILL_ORDER if not skill_successes.get(sk, False)), None)

    # Average steps for successful gather skills
    succ_gather = [r["steps"] for sk, r in result["results"].items()
                   if sk in GATHER_SKILLS and r.get("success")]
    avg_succ_steps = sum(succ_gather) / len(succ_gather) if succ_gather else 0.0

    return {
        "frac_stuck": frac_stuck,
        "n_gather_attempted": len(gather_recs),
        "n_gather_stuck": len(stuck),
        "skill_successes": skill_successes,
        "first_fail": first_fail,
        "avg_succ_gather_steps": round(avg_succ_steps, 1),
    }


# ---------------------------------------------------------------------------
# Tuning decision
# ---------------------------------------------------------------------------

def pick_tuning(result: Dict[str, Any], diag: Dict[str, Any]) -> Dict[str, Any]:
    """Return tuning params + reason for the next iteration."""
    for rule in TUNING_RULES:
        if rule["condition"](result, diag):
            params = {**DEFAULTS, **rule["params"]}
            return {"rule": rule["name"], "params": params, "reason": rule["reason"]}
    # Already at stretch — no further tuning needed
    return {
        "rule": "none",
        "params": DEFAULTS.copy(),
        "reason": "Diamond reached — no further tuning required",
    }


# ---------------------------------------------------------------------------
# Pretty report
# ---------------------------------------------------------------------------

TIER_LABELS = {
    TIER_BASELINE:   "BASELINE",
    TIER_ACCEPTABLE: "ACCEPTABLE",
    TIER_TARGET:     "TARGET",
    TIER_STRETCH:    "STRETCH",
}

def tier_reached(result: Dict[str, Any]) -> int:
    if result["diamond_reached"]:
        return TIER_STRETCH
    for t in (TIER_TARGET, TIER_ACCEPTABLE, TIER_BASELINE):
        if result["n_achieved"] >= t:
            return t
    return 0


def print_report(result: Dict[str, Any], diag: Dict[str, Any], tune: Dict[str, Any]) -> None:
    t = tier_reached(result)
    tier_str = TIER_LABELS.get(t, "NONE")

    print("\n" + "=" * 62)
    print(f"  DIA+GROOT Eval Report")
    print(f"  mode={result['mode']}  seed={result['seed']}  "
          f"elapsed={result['elapsed_s']}s")
    print("=" * 62)

    print(f"\n  Performance tier : {tier_str}  (n_achieved={result['n_achieved']}/9)")
    print(f"  diamond_reached  : {result['diamond_reached']}")
    print(f"  total_steps      : {result['total_steps']}")

    print(f"\n  Skill results:")
    for sk in SKILL_ORDER:
        rec = result["results"].get(sk, {})
        icon  = "✓" if rec.get("success") else "✗"
        steps = rec.get("steps", "-")
        extra = f"  [{rec.get('craft_method','')}]" if sk in CRAFT_SKILLS else ""
        print(f"    {icon}  {sk:<15}  steps={steps:<6}{extra}")

    print(f"\n  Diagnostics:")
    print(f"    frac_stuck           = {diag['frac_stuck']:.2f}  "
          f"({diag['n_gather_stuck']}/{diag['n_gather_attempted']} gather skills at cap)")
    print(f"    avg_succ_gather_steps= {diag['avg_succ_gather_steps']}")
    print(f"    first_fail           = {diag['first_fail']}")

    print(f"\n  Next tuning rule : {tune['rule']}")
    print(f"  Reason           : {tune['reason']}")
    print(f"  Next params      : {tune['params']}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Next-run command
# ---------------------------------------------------------------------------

def next_run_cmd(tune: Dict[str, Any], seed: int, iteration: int) -> str:
    p = tune["params"]
    out = f"/tmp/dia_groot_ft_s{seed}_it{iteration}.json"
    log = f"/tmp/dia_groot_ft_s{seed}_it{iteration}.log"
    cmd = (
        f"PYTHONUNBUFFERED=1 conda run -n dia-minecraft --no-capture-output "
        f"python -u scripts/run_groot_dia_minestudio.py "
        f"--mode {p['mode']} "
        f"--seed {seed} "
        f"--max_steps_per_skill {p['max_steps_per_skill']} "
        f"--max_total_steps {p['max_total_steps']} "
        f"--out {out} "
        f"> {log} 2>&1"
    )
    return cmd, out, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_result() -> Optional[Path]:
    candidates = sorted(
        Path("/tmp").glob("dia_groot_ft_s*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    ) + sorted(
        EVAL_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return candidates[0] if candidates else None


def load_result(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def log_iteration(result, tune, iteration, result_path):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "iteration": iteration,
        "result_path": str(result_path),
        "n_achieved": result["n_achieved"],
        "diamond_reached": result["diamond_reached"],
        "tier": tier_reached(result),
        "tune_rule": tune["rule"],
        "tune_params": tune["params"],
    }
    with open(TUNE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval-and-tune DIA+GROOT results")
    ap.add_argument("--result",   default=None, help="Path to result JSON")
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--iter",     type=int, default=1, help="Iteration number")
    ap.add_argument("--launch",   action="store_true",
                    help="Auto-launch next tuned run in background")
    args = ap.parse_args()

    # 1. Load result
    result_path = Path(args.result) if args.result else find_latest_result()
    if result_path is None or not result_path.exists():
        print("No result JSON found. Run the pilot first:", file=sys.stderr)
        print("  conda run -n dia-minecraft python -u scripts/run_groot_dia_minestudio.py "
              "--mode dia --seed 0 --out /tmp/dia_groot_ft_s0.json", file=sys.stderr)
        sys.exit(1)
    result = load_result(result_path)

    # 2. Run pytest eval suite against this result
    print(f"\nRunning eval suite against: {result_path}")
    pytest_cmd = [
        "conda", "run", "-n", "dia-minecraft",
        "python", "-m", "pytest", "tests/test_pilot_eval.py",
        "--result", str(result_path),
        "-v", "--tb=short",
        "-p", "no:launch_ros", "-p", "no:launch_testing",
    ]
    subprocess.run(pytest_cmd, cwd=str(REPO_ROOT))

    # 3. Diagnose + pick tuning
    diag = diagnose(result)
    tune = pick_tuning(result, diag)
    print_report(result, diag, tune)

    # 4. Log iteration
    log_iteration(result, tune, args.iter, result_path)

    # 5. Emit / launch next command
    if tune["rule"] == "none":
        print("\n  >>> Requirements met — no further tuning needed. <<<\n")
        return

    cmd, out_path, log_path = next_run_cmd(tune, seed=args.seed, iteration=args.iter + 1)
    print(f"\n  Next run command:\n  {cmd}\n")

    if args.launch:
        print(f"  Launching in background → log: {log_path}")
        subprocess.Popen(cmd, shell=True, cwd=str(REPO_ROOT))
        print("  Launched.")


if __name__ == "__main__":
    main()
