#!/usr/bin/env python3
# scripts/dia_cli.py
from __future__ import annotations

import argparse
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPT_MAP = {
    "minecraft2d": "train_minecraft2d_dia.py",
    "coinrun": "train_coinrun_dia.py",
    "causalworld": "train_causalworld_dia.py",
    "cartpole": "train_cartpole_dia.py",
    "montezuma": "train_montezuma_dia.py",
}

def run(args):
    env = args.env.lower()
    if env not in SCRIPT_MAP:
        raise SystemExit(f"Unknown env '{env}'. Run `scripts/dia_cli.py list` to see options.")
    script = os.path.join(HERE, SCRIPT_MAP[env])
    cmd = [sys.executable, script,
           "--pcg", args.pcg,
           "--steps", str(args.steps),
           "--fit_every", str(args.fit_every),
           "--pcg_epochs", str(args.pcg_epochs),
           "--buffer_recent", str(args.buffer_recent),
           "--min_buffer", str(args.min_buffer),
           "--logdir", args.logdir]
    if args.goal:
        cmd.extend(["--task_goal", args.goal])
    if args.env_id:
        cmd.extend(["--env_id", args.env_id])
    if args.use_ppo_options:
        cmd.append("--use_ppo_options")
    if args.train_options:
        cmd.append("--train_options")
    if args.ppo_steps:
        cmd.extend(["--ppo_steps", str(args.ppo_steps)])

    print(">>>", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))

def _add_common(p):
    p.add_argument("--pcg", default="notears", choices=["notears", "variational", "simple"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--fit_every", type=int, default=10)
    p.add_argument("--pcg_epochs", type=int, default=200)
    p.add_argument("--buffer_recent", type=int, default=1024)
    p.add_argument("--min_buffer", type=int, default=256)
    p.add_argument("--logdir", type=str, default="runs/dia")
    p.add_argument("--goal", type=str, default=None, help="task goal variable (e.g., diamond, coin_collected)")
    p.add_argument("--env_id", type=str, default=None, help="override underlying gym env id if supported")
    p.add_argument("--use_ppo_options", action="store_true")
    p.add_argument("--train_options", action="store_true")
    p.add_argument("--ppo_steps", type=int, default=3000)

def main():
    ap = argparse.ArgumentParser(prog="dia_cli", description="DIA unified CLI")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p_run = sp.add_parser("run", help="Run a DIA training script")
    p_run.add_argument("--env", required=True, choices=list(SCRIPT_MAP.keys()))
    _add_common(p_run)
    p_run.set_defaults(func=run)

    p_list = sp.add_parser("list", help="List available environments/backends")
    def _list(_):
        print("Environments:", ", ".join(SCRIPT_MAP.keys()))
        print("PCG backends: notears | variational | simple")
        print("Examples:")
        print("  python scripts/dia_cli.py run --env minecraft2d --pcg notears --steps 300 --goal diamond")
        print("  python scripts/dia_cli.py run --env coinrun --pcg variational --steps 300 --goal coin_collected")
        print("  python scripts/dia_cli.py run --env montezuma --pcg notears --steps 200 --goal has_key")
        sys.exit(0)
    p_list.set_defaults(func=_list)

    args = ap.parse_args()
    return args.func(args)

if __name__ == "__main__":
    main()
