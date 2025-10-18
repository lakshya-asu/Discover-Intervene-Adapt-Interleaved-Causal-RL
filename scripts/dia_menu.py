#!/usr/bin/env python3
# scripts/dia_menu.py
from __future__ import annotations

import subprocess
import sys

OPTS = {
    "envs": ["minecraft2d", "coinrun", "cartpole", "causalworld", "montezuma"],
    "pcg": ["notears", "variational", "simple"],
}

DEFAULTS = {
    "env": "minecraft2d",
    "pcg": "notears",
    "steps": 300,
    "fit_every": 10,
    "pcg_epochs": 200,
    "buffer_recent": 1024,
    "min_buffer": 256,
    "goal_by_env": {
        "minecraft2d": "diamond",
        "coinrun": "coin_collected",
        "cartpole": "x",
        "causalworld": "tower_height",
        "montezuma": "has_key",
    },
    "logdir": "runs/menu",
}

def ask(prompt, default=None, cast=str):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    return cast(raw)

def choose(label, options, default):
    print(f"\nSelect {label}:")
    for i, v in enumerate(options, 1):
        print(f"  {i}. {v}")
    while True:
        idx = ask(f"{label} number", default=str(options.index(default) + 1), cast=str)
        if idx.isdigit() and 1 <= int(idx) <= len(options):
            return options[int(idx) - 1]
        print("Invalid choice; try again.")

def launch(env, pcg, steps, fit_every, pcg_epochs, buffer_recent, min_buffer, goal, logdir, env_id=None):
    cmd = [
        sys.executable, "scripts/dia_cli.py", "run",
        "--env", env,
        "--pcg", pcg,
        "--steps", str(steps),
        "--fit_every", str(fit_every),
        "--pcg_epochs", str(pcg_epochs),
        "--buffer_recent", str(buffer_recent),
        "--min_buffer", str(min_buffer),
        "--logdir", logdir,
        "--goal", goal,
    ]
    if env_id:
        cmd += ["--env_id", env_id]
    print("\n>>>", " ".join(cmd))
    return subprocess.call(cmd)

def main():
    print("=== DIA Menu ===")
    while True:
        env = choose("environment", OPTS["envs"], DEFAULTS["env"])
        pcg = choose("PCG backend", OPTS["pcg"], DEFAULTS["pcg"])
        steps = ask("steps", DEFAULTS["steps"], int)
        fit_every = ask("fit_every", DEFAULTS["fit_every"], int)
        pcg_epochs = ask("pcg_epochs", DEFAULTS["pcg_epochs"], int)
        buffer_recent = ask("buffer_recent", DEFAULTS["buffer_recent"], int)
        min_buffer = ask("min_buffer", DEFAULTS["min_buffer"], int)
        goal = ask("goal variable", DEFAULTS["goal_by_env"][env], str)
        logdir = ask("logdir", f"{DEFAULTS['logdir']}/{env}", str)
        env_id = None
        if env in ("cartpole", "montezuma"):
            env_id = ask("env_id (optional)", "", str)
            if env_id == "":
                env_id = None

        rc = launch(env, pcg, steps, fit_every, pcg_epochs, buffer_recent, min_buffer, goal, logdir, env_id)
        print(f"\nRun finished with exit code {rc}")
        again = ask("Run another? (y/n)", "y", str).lower()
        if not again.startswith("y"):
            break
    print("Done.")

if __name__ == "__main__":
    main()
