from __future__ import annotations
import numpy as np
from gymnasium import spaces
from procgen import ProcgenGym3Env
from .base import EnvAPI

class ProcgenCoinRunEnv(EnvAPI):
    """Wrap ProcgenGym3Env with a Gymnasium-style single-env API."""

    def __init__(self, start_level=0, num_levels=1):
        self._env = ProcgenGym3Env(
            num=1, env_name="coinrun", start_level=start_level, num_levels=num_levels
        )
        self._observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self._action_space = spaces.Discrete(15)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        # gym3 Procgen has no .seed(); for determinism use start_level/num_levels at init.
        obs = self._env.reset()
        rgb = obs["rgb"][0]
        info_list = self._env.get_info()
        info = info_list[0] if info_list else {}
        return rgb, info

    def step(self, action):
        act = np.array([action], dtype=np.int32)
        self._env.step(act)
        obs = self._env.observe()
        rew = float(self._env.get_reward()[0])
        done = bool(self._env.get_done()[0])
        rgb = obs["rgb"][0]
        info_list = self._env.get_info()
        info = info_list[0] if info_list else {}
        terminated = done
        truncated = False
        return rgb, rew, terminated, truncated, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
