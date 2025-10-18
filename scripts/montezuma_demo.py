# scripts/montezuma_demo.py
#!/usr/bin/env python3
"""
Small demo: wrap Montezuma, build EVGS, and print detected variables for a few steps.

Requires: gym[atari] or gymnasium[atari] and ALE ROMs installed.
"""

from __future__ import annotations

import time
import numpy as np

try:
    import gym
except Exception:
    gym = None

from dia.evgs_adapters import make_montezuma_evgs
from dia.evgs_montezuma import wrap_montezuma_env, MontezumaDetectorsConfig


def main():
    if gym is None:
        raise RuntimeError("Please install gym/gymnasium with Atari support to run this demo.")

    # You may use different IDs depending on your Gym version/wrappers
    env_id = "MontezumaRevengeNoFrameskip-v4"
    env = gym.make(env_id)

    # Configure detectors (RAM addresses left None by default; pixel fallback will be used)
    cfg = MontezumaDetectorsConfig(
        room_addr=None,
        has_key_addr=None,
        door_addr=None,
        player_x_addr=None, player_y_addr=None,
        skull_x_addr=None, skull_y_addr=None,
    )

    env = wrap_montezuma_env(env, cfg)
    evgs = make_montezuma_evgs()

    obs = env.reset()
    print("Initial EV vector:", evgs.extract(obs), "| info:", obs.get("info", {}))

    for t in range(50):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        x = evgs.extract(obs)
        if t % 5 == 0:
            print(f"t={t:03d} EV={x} info={{has_key:{info.get('has_key')}, door_open:{info.get('door_open')}, room_id:{info.get('room_id')}, skull_near:{info.get('skull_near')}}}")
        if done:
            obs = env.reset()

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
