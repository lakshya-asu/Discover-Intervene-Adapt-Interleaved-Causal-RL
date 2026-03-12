# src/dia/options_coinrun.py
"""
PixelStackPPOOption: PPO option for CoinRun (or any pixel env wrapped in DIA).

Handles:
  - Nested dict obs unwrapping (DIA env returns {"obs": {"obs": array, ...}, ...})
  - Rolling frame stack (n_stack frames concatenated along channels, matching
    the VecFrameStack format used during training)
  - SB3 model.predict() inference
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np

from .options import OptionPolicy, OptionConfig
from .types import Subgoal
from .evgs import EVGS

try:
    from stable_baselines3 import PPO
    SB3_OK = True
except Exception:
    PPO = None
    SB3_OK = False


def _unwrap_pixel(obs: Any) -> np.ndarray:
    """Peel {"obs": {"obs": array}} nesting until we reach an ndarray."""
    raw = obs
    for _ in range(8):
        if isinstance(raw, np.ndarray):
            break
        if isinstance(raw, dict):
            raw = raw.get("obs", raw)
        else:
            break
    if not isinstance(raw, np.ndarray):
        raise ValueError(f"Could not extract ndarray from obs type {type(obs)}")
    return raw.astype(np.uint8)


class PixelStackPPOOption(OptionPolicy):
    """
    Option backed by a CNN PPO model trained with VecFrameStack(n_stack).
    Call reset_stack() between episodes; it is called automatically in run().
    """

    def __init__(
        self,
        subgoal: Subgoal,
        cfg: OptionConfig,
        model,
        n_stack: int = 4,
    ):
        super().__init__(subgoal, cfg)
        self.model  = model
        self.n_stack = int(n_stack)
        self._buf: Optional[np.ndarray] = None  # (H, W, C * n_stack) uint8

    def reset_stack(self):
        self._buf = None

    def _push(self, frame: np.ndarray):
        """Push one (H, W, C) frame into the rolling buffer."""
        h, w, c = frame.shape
        if self._buf is None:
            self._buf = np.zeros((h, w, c * self.n_stack), dtype=np.uint8)
            # Prime buffer: fill all slots with this frame
            for i in range(self.n_stack):
                self._buf[..., i * c:(i + 1) * c] = frame
        else:
            # Shift oldest out, push newest in (channels-last ordering)
            self._buf = np.roll(self._buf, -c, axis=2)
            self._buf[..., -c:] = frame

    def act(self, obs: Any) -> int:
        frame = _unwrap_pixel(obs)
        self._push(frame)
        action, _ = self.model.predict(self._buf, deterministic=True)
        # predict() returns scalar or 1-d array depending on context
        return int(action) if np.ndim(action) == 0 else int(action.flat[0])

    def run(self, env, evgs: EVGS) -> dict:
        """Execute option until success or horizon; returns DIA-compatible record."""
        self.reset_stack()

        obs   = env.get_obs() if hasattr(env, "get_obs") else env.reset()
        frame = _unwrap_pixel(obs)
        self._push(frame)   # prime stack with first obs

        x_t   = evgs.extract(obs)
        steps = 0
        success   = False
        trajectory = []

        while steps < self.cfg.max_steps:
            action = self.act(obs)
            next_obs, _rew, done, info = env.step(action)
            x_tp1     = evgs.extract(next_obs)
            succ_this = EVGS.predicate_holds(x_t, x_tp1, self.subgoal)
            trajectory.append((obs, action, next_obs, succ_this))
            obs = next_obs
            steps += 1

            if succ_this:
                success = True
                if self.cfg.terminate_on_success:
                    break
            if done and not info.get("soft_continue", False):
                break
            x_t = x_tp1

        return {
            "success":   success,
            "steps":     steps,
            "trajectory": trajectory,
            "final_obs": obs,
        }


def load_coinrun_ppo_option(
    path: str,
    subgoal: Subgoal,
    cfg: Optional[OptionConfig] = None,
    n_stack: int = 4,
) -> PixelStackPPOOption:
    """Load a saved SB3 PPO model and wrap it in a PixelStackPPOOption."""
    if not SB3_OK:
        raise RuntimeError("stable_baselines3 not installed")
    if cfg is None:
        cfg = OptionConfig(max_steps=512, terminate_on_success=True)
    model = PPO.load(path)
    return PixelStackPPOOption(subgoal=subgoal, cfg=cfg, model=model, n_stack=n_stack)
