#!/usr/bin/env python3
"""
scripts/watch_coinrun.py — Record CoinRun gameplay to MP4.

In ProcGen, the observation IS the rendered 64x64x3 RGB frame,
so we capture obs directly without needing a display.

Usage:
    conda activate dia
    cd Discover-Intervene-Adapt-Interleaved-Causal-RL
    python scripts/watch_coinrun.py --episodes 3 --out coinrun.mp4
"""
from __future__ import annotations

import argparse
import numpy as np
import imageio

import gym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--out", type=str, default="coinrun.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=4,
                    help="Upscale factor for 64x64 frames (default 4 → 256x256)")
    args = ap.parse_args()

    env = gym.make(
        "procgen:procgen-coinrun-v0",
        num_levels=0,
        start_level=args.seed,
    )

    frames = []
    total_reward = 0.0

    def add_frame(obs):
        if not (isinstance(obs, np.ndarray) and obs.ndim == 3):
            return
        # Upscale via nearest-neighbor for visibility
        if args.scale > 1:
            obs = obs.repeat(args.scale, axis=0).repeat(args.scale, axis=1)
        frames.append(obs.astype(np.uint8))

    print(f"Recording {args.episodes} episode(s), up to {args.max_steps} steps each…")
    for ep in range(args.episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        add_frame(obs)

        ep_reward = 0.0
        for step in range(args.max_steps):
            action = env.action_space.sample()
            step_result = env.step(action)
            obs, reward = step_result[0], step_result[1]
            done = step_result[2] if len(step_result) == 4 else (step_result[2] or step_result[3])
            ep_reward += float(reward)
            add_frame(obs)
            if done:
                break

        total_reward += ep_reward
        print(f"  ep {ep+1}: {step+1} steps, reward={ep_reward:.1f}")

    env.close()

    if not frames:
        print("ERROR: no frames captured.")
        return

    print(f"Writing {len(frames)} frames ({frames[0].shape}) → {args.out}  (fps={args.fps})")
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8) as writer:
        for f in frames:
            writer.append_data(f)

    print(f"Done. Total reward across {args.episodes} episodes: {total_reward:.1f}")
    print(f"Video: {args.out}")


if __name__ == "__main__":
    main()
