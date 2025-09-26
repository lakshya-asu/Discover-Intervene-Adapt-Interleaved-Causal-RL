"""Gymnasium-compatible wrapper for Procgen CoinRun using Gym3."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - prefer Gymnasium API
    import gymnasium as _gym
    BaseEnv = _gym.Env
    spaces = _gym.spaces
except ImportError:  # pragma: no cover - fallback to Gym
    try:
        import gym as _gym  # type: ignore
        BaseEnv = _gym.Env  # type: ignore[attr-defined]
        spaces = _gym.spaces  # type: ignore
    except ImportError:
        _gym = None  # type: ignore
        BaseEnv = object
        spaces = None  # type: ignore

try:  # pragma: no cover - core dependency
    from procgen import ProcgenGym3Env
except ImportError as exc:  # pragma: no cover - handled at instantiation
    ProcgenGym3Env = None  # type: ignore
    _PROCGEN_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency available
    _PROCGEN_IMPORT_ERROR = None


_FRIENDLY_PROCGEN_HINT = (
    "Procgen is not installed. Install it with `pip install procgen2 gymnasium shimmy`."
)
_OBS_SHAPE = (64, 64, 3)
_NUM_ACTIONS = 15


class ProcgenCoinRunEnv(BaseEnv):
    """Single-environment Procgen CoinRun wrapper exposing Gymnasium semantics."""

    metadata = {"render_modes": ["rgb_array"]}
    NUM_ACTIONS = _NUM_ACTIONS

    def __init__(
        self,
        *,
        start_level: int = 0,
        num_levels: int = 0,
        distribution_mode: str = "easy",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if ProcgenGym3Env is None:
            raise ImportError(_FRIENDLY_PROCGEN_HINT) from _PROCGEN_IMPORT_ERROR
        if spaces is None:
            raise ImportError("Gymnasium or Gym is required for ProcgenCoinRunEnv.")

        self._env = ProcgenGym3Env(
            num=1,
            env_name="coinrun",
            start_level=start_level,
            num_levels=num_levels,
            distribution_mode=distribution_mode,
            render_mode=render_mode or "rgb_array",
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=_OBS_SHAPE,
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(_NUM_ACTIONS)
        self._last_info: Dict[str, Any] = {}
        self._start_level = start_level
        self._num_levels = num_levels

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._env.seed(seed)
        self._env.reset()
        obs = self._extract_observation()
        info = self._extract_info()
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        np_action = np.asarray([action], dtype=np.int32)
        obs, reward, done, info = self._env.step(np_action)
        obs0 = self._format_observation(obs[0])
        reward0 = float(np.asarray(reward)[0])
        done0 = bool(np.asarray(done)[0])
        info0 = self._format_info(info[0])
        self._last_info = info0
        return obs0, reward0, done0, False, info0

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._env.close()

    @property
    def level_seed(self) -> Optional[int]:
        return self._last_info.get("level_seed")

    def get_factors(self) -> Dict[str, Any]:
        return dict(self._last_info)

    # ------------------------------------------------------------------
    def _extract_observation(self) -> np.ndarray:
        obs = self._env.observe()
        return self._format_observation(obs[0])

    def _extract_info(self) -> Dict[str, Any]:
        info = self._env.get_info()[0]
        info = self._format_info(info)
        self._last_info = info
        return info

    def _format_observation(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.uint8)
        if arr.shape != _OBS_SHAPE:
            arr = arr.reshape(_OBS_SHAPE)
        return arr

    def _format_info(self, info: Any) -> Dict[str, Any]:
        if isinstance(info, dict):
            return dict(info)
        return {}


__all__ = ["ProcgenCoinRunEnv"]
