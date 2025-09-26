"""Stable-Baselines3 PPO training helpers."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional, Union

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import make_vec_env
except ImportError as exc:  # pragma: no cover - inform user
    PPO = None  # type: ignore
    make_vec_env = None  # type: ignore
    _SB3_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency available
    _SB3_IMPORT_ERROR = None

try:  # pragma: no cover - spaces import for observation inspection
    from gymnasium import spaces as gym_spaces
except ImportError:  # pragma: no cover - fallback to Gym
    try:
        from gym import spaces as gym_spaces  # type: ignore
    except ImportError:
        gym_spaces = None  # type: ignore

if "gym_spaces" in locals() and gym_spaces is not None:
    DictSpace = getattr(gym_spaces, "Dict", None)  # type: ignore[attr-defined]
else:
    DictSpace = None


_FRIENDLY_SB3_HINT = (
    "Stable-Baselines3 is not installed. Install it with "
    "`pip install stable-baselines3[extra]`."
)


def train(env_name: str, total_steps: int, seed: Optional[int] = None) -> Any:
    """Train PPO on the provided environment identifier.

    Args:
        env_name: Gym environment ID or `module:Constructor` path for a callable
            returning a Gym-compatible environment instance.
        total_steps: Total training timesteps; must be positive.
        seed: Optional RNG seed applied to PPO and environment creation.

    Returns:
        The trained PPO model.
    """

    if PPO is None or make_vec_env is None:
        raise ImportError(_FRIENDLY_SB3_HINT) from _SB3_IMPORT_ERROR
    if total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")

    env_factory = _resolve_env_factory(env_name)
    vec_env = make_vec_env(env_factory, n_envs=1, seed=seed)
    try:
        policy = "MlpPolicy"
        observation_space = getattr(vec_env, "observation_space", None)
        if DictSpace is not None and isinstance(observation_space, DictSpace):  # type: ignore[arg-type]
            policy = "MultiInputPolicy"
        model = PPO(policy, vec_env, seed=seed, verbose=1)
        model.learn(total_timesteps=total_steps)
    finally:
        vec_env.close()
    return model


def _resolve_env_factory(env_name: str) -> Union[str, Callable[[], Any]]:
    """Return a make-able env identifier or factory callable for SB3."""

    if ":" not in env_name:
        return env_name

    module_path, attr_name = env_name.split(":", 1)
    if not module_path or not attr_name:
        raise ValueError(
            "Custom environment name must be in 'module:callable' format"
        )

    module = importlib.import_module(module_path)
    env_ctor = getattr(module, attr_name)
    if not callable(env_ctor):
        raise TypeError(f"Resolved object {env_ctor!r} is not callable")

    def _factory() -> Any:
        return env_ctor()

    return _factory


__all__ = ["train"]
