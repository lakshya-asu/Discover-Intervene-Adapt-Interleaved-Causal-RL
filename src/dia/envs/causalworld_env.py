"""Gymnasium-compatible wrapper for the CausalWorld pushing task."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import Gymnasium base if available
    import gymnasium as _gym
    BaseEnv = _gym.Env
except ImportError:  # pragma: no cover - fallback to Gym
    try:
        import gym as _gym  # type: ignore
        BaseEnv = _gym.Env  # type: ignore[attr-defined]
    except ImportError:
        _gym = None  # type: ignore
        BaseEnv = object

try:  # pragma: no cover - optional dependency
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback for older Gym installs
    from gym import spaces  # type: ignore

try:  # pragma: no cover - import guard for causal_world
    import causal_world  # noqa: F401  # ensure package is available
    from causal_world.envs.causalworld import CausalWorld  # type: ignore
    from causal_world.task_generators.task_generator import task_generator  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at instantiation time
    CausalWorld = None  # type: ignore
    task_generator = None  # type: ignore
    _CAUSALWORLD_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency available
    _CAUSALWORLD_IMPORT_ERROR = None


_FRIENDLY_IMPORT_HINT = (
    "CausalWorld is not installed. Install it with `pip install causalworld`."
)
_RGB_SHAPE = (64, 64, 3)


class CausalWorldPushingEnv(BaseEnv):
    """Wrap the CausalWorld pushing benchmark with Gymnasium-style APIs."""

    metadata = {"render_modes": ["rgb_array"]}

    _FACTOR_PATHS: Dict[str, Sequence[Sequence[str]]] = {
        "block_mass": (
            ("task", "block", "mass"),
            ("block", "mass"),
            ("mass",),
        ),
        "block_size": (
            ("task", "block", "size"),
            ("block", "size"),
            ("size",),
            ("block", "half_size"),
        ),
        "friction": (
            ("world", "surface", "friction"),
            ("table", "friction"),
            ("friction",),
        ),
    }

    def __init__(
        self,
        observation_mode: str = "structured",
        action_mode: Optional[str] = None,
        skip_frame: int = 1,
    ) -> None:
        super().__init__()
        if CausalWorld is None or task_generator is None:
            raise ImportError(_FRIENDLY_IMPORT_HINT) from _CAUSALWORLD_IMPORT_ERROR

        self._task = task_generator(task_generator_id="pushing")
        self._env = CausalWorld(
            task=self._task,
            observation_mode=observation_mode,
            action_mode=action_mode or "joint_torques",
            enable_visualization=False,
            skip_frame=skip_frame,
        )

        self._last_factors: Dict[str, Any] = {k: None for k in self._FACTOR_PATHS}
        self._proprio_dim = int(
            np.prod(getattr(self._env.observation_space, "shape", ()) or (0,))
        )
        proprio_space = (
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._proprio_dim,),
                dtype=np.float32,
            )
            if self._proprio_dim > 0
            else None
        )
        obs_spaces = {
            "rgb": spaces.Box(low=0, high=255, shape=_RGB_SHAPE, dtype=np.uint8),
        }
        if proprio_space is not None:
            obs_spaces["proprio"] = proprio_space
        self._observation_space = spaces.Dict(obs_spaces)

    # ------------------------------------------------------------------
    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            seed_fn = getattr(self._env, "seed", None)
            if callable(seed_fn):  # pragma: no branch - defensive
                seed_fn(seed)

        if options and "factors" in options:
            self.intervene(options["factors"])

        raw_obs = self._env.reset()
        self._refresh_factor_cache()
        obs = self._compose_observation(raw_obs)
        info: Dict[str, Any] = {"factors": self.get_factors()}
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        raw_obs, reward, done, info = self._env.step(action)
        self._refresh_factor_cache()
        info = dict(info)
        info.setdefault("factors", self.get_factors())
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
        return (
            self._compose_observation(raw_obs),
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ------------------------------------------------------------------
    def get_factors(self) -> Dict[str, Any]:
        return {k: v for k, v in self._last_factors.items() if v is not None}

    def intervene(self, spec: Dict[str, Any]) -> bool:
        if not spec:
            return False

        do_intervention = getattr(self._env, "do_intervention", None)
        if not callable(do_intervention):
            return False
        try:  # pragma: no cover - backend specific
            do_intervention(spec)
        except Exception:
            return False
        self._last_factors.update(spec)
        return True

    def close(self) -> None:
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()

    # ------------------------------------------------------------------
    def _compose_observation(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        proprio = None
        if raw_obs is not None and self._proprio_dim > 0:
            proprio_arr = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
            if proprio_arr.size != self._proprio_dim:
                proprio = proprio_arr.astype(np.float32)
            else:
                proprio = proprio_arr

        rgb = self._capture_rgb()
        obs = {"rgb": rgb}
        if proprio is not None:
            obs["proprio"] = proprio
        return obs

    def _capture_rgb(self) -> np.ndarray:
        render_fn = getattr(self._env, "render", None)
        if callable(render_fn):
            try:
                frame = render_fn(
                    mode="rgb_array", width=_RGB_SHAPE[1], height=_RGB_SHAPE[0]
                )
            except TypeError:
                try:  # pragma: no cover - alternative signature
                    frame = render_fn(mode="rgb_array")
                except Exception:
                    frame = None
            except Exception:  # pragma: no cover - backend specific
                frame = None
        else:
            frame = None

        if frame is None:
            return np.zeros(_RGB_SHAPE, dtype=np.uint8)

        arr = np.asarray(frame, dtype=np.uint8)
        if arr.shape != _RGB_SHAPE:
            try:
                arr = arr.reshape(_RGB_SHAPE)
            except ValueError:
                arr = np.zeros(_RGB_SHAPE, dtype=np.uint8)
        return arr

    def _refresh_factor_cache(self) -> None:
        sources: Iterable[Optional[Dict[str, Any]]] = (
            self._call_safely(self._env, "get_task_state"),
            self._call_safely(self._env, "get_latent_state"),
            self._call_safely(self._task, "get_current_randomization_parameters"),
        )
        for source in sources:
            if isinstance(source, dict):
                self._update_factor_cache_from_dict(source)

    def _call_safely(self, obj: Any, name: str) -> Optional[Dict[str, Any]]:
        attr = getattr(obj, name, None)
        if not callable(attr):
            return None
        try:  # pragma: no cover - backend specific
            result = attr()
        except Exception:
            return None
        return result if isinstance(result, dict) else None

    def _update_factor_cache_from_dict(self, data: Dict[str, Any]) -> None:
        for factor, paths in self._FACTOR_PATHS.items():
            value = None
            for path in paths:
                value = self._nested_get(data, path)
                if value is not None:
                    break
            if value is None:
                value = self._search_by_suffix(data, paths)
            if value is not None:
                self._last_factors[factor] = value

    def _nested_get(self, data: Dict[str, Any], path: Sequence[str]) -> Any:
        current: Any = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _search_by_suffix(
        self,
        data: Dict[str, Any],
        paths: Sequence[Sequence[str]],
    ) -> Any:
        suffixes = {path[-1] for path in paths if path}
        stack = [data]
        while stack:
            current = stack.pop()
            if not isinstance(current, dict):
                continue
            for key, value in current.items():
                if key in suffixes:
                    return value
                if isinstance(value, dict):
                    stack.append(value)
        return None


__all__ = ["CausalWorldPushingEnv"]
