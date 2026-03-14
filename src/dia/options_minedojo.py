# src/dia/options_minedojo.py
"""
PPO-based skill options for MineDojo 3D Minecraft.

Each skill (PPO model) is trained separately via scripts/train_minedojo_skill.py,
then loaded here for execution during Phase 2 transfer.

Architecture
------------
MinedojoPPOOption
  - Wraps a pre-trained SB3 PPO model
  - Preprocesses MineDojo's multimodal obs → compact (image + inventory) feature
  - Runs until subgoal achieved (var increases) or max_steps
  - Same .run(env, evgs) interface as CoinRun PixelStackPPOOption

MinedojoObsWrapper  (gym.Wrapper)
  - Converts raw MineDojo obs dict → compact obs for SB3 training
  - Outputs Dict space: {"rgb": Box(H,W,3), "inventory": Box(9,)}
  - Used during training AND inference

ItemRewardWrapper  (gym.RewardWrapper)
  - Adds +1 reward when target inventory item count increases
  - Used during PPO skill training

Observation design for SB3
---------------------------
  "rgb":       (64, 64, 3) uint8  — downsampled from MineDojo's 160×256
  "inventory": (9,)       float32 — DIA variable vector from evgs_minedojo

SB3 MultiInputPolicy handles this Dict space with:
  - CnnPolicy branch for "rgb"
  - MlpPolicy branch for "inventory"
  Both branches are concatenated before the action head.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

from .types import Subgoal, Predicate
from .evgs import EVGS
from .options import OptionPolicy, OptionConfig

try:
    import gym
    from gym import spaces
    GYM_OK = True
except Exception:
    gym = None; spaces = None; GYM_OK = False

try:
    from stable_baselines3 import PPO as SB3PPO
    SB3_OK = True
except Exception:
    SB3PPO = None; SB3_OK = False


# ── Observation constants ─────────────────────────────────────────────────────
OBS_H  = 64    # RGB height fed to CNN
OBS_W  = 64    # RGB width  fed to CNN
INV_DIM = 9    # DIA variable vector length


# ---------------------------------------------------------------------------
# MinedojoObsWrapper — convert raw MineDojo obs to SB3-compatible dict
# ---------------------------------------------------------------------------

class MinedojoObsWrapper(gym.Wrapper if GYM_OK else object):
    """
    Wraps a MineDojo environment.

    Input:  raw MineDojo obs dict with 'rgb', 'inventory', ...
    Output: {'rgb': (64,64,3) uint8, 'inventory': (9,) float32}

    Also exposes get_obs() for compatibility with OptionPolicy.run().
    """

    def __init__(self, env, evgs: EVGS):
        if GYM_OK:
            super().__init__(env)
        else:
            self.env = env
        self.evgs = evgs
        self._last_obs: Any = None

        if GYM_OK and spaces is not None:
            self.observation_space = spaces.Dict({
                "rgb":       spaces.Box(0, 255, (OBS_H, OBS_W, 3), dtype=np.uint8),
                "inventory": spaces.Box(0.0, 1.0, (INV_DIM,), dtype=np.float32),
            })

    def _convert(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        rgb = self._get_rgb(raw_obs)
        inv = self.evgs.extract(raw_obs).astype(np.float32)
        return {"rgb": rgb, "inventory": inv}

    @staticmethod
    def _get_rgb(obs: Any) -> np.ndarray:
        """Extract and resize RGB frame from MineDojo obs."""
        try:
            import cv2
            img = np.asarray(obs["rgb"], dtype=np.uint8)
            if img.ndim == 3:
                return cv2.resize(img, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
        # Fallback: black frame
        return np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8)

    def reset(self, **kwargs):
        raw = self.env.reset(**kwargs)
        raw = raw[0] if isinstance(raw, tuple) else raw
        self._last_obs = raw
        return self._convert(raw)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            raw, rew, term, trunc, info = result
            done = term or trunc
        else:
            raw, rew, done, info = result
        self._last_obs = raw
        return self._convert(raw), float(rew), bool(done), info

    def get_obs(self) -> Dict[str, np.ndarray]:
        if self._last_obs is not None:
            return self._convert(self._last_obs)
        return {"rgb": np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8),
                "inventory": np.zeros(INV_DIM, dtype=np.float32)}

    @property
    def action_space(self):
        return self.env.action_space


# ---------------------------------------------------------------------------
# ItemRewardWrapper — shape reward for skill training
# ---------------------------------------------------------------------------

class ItemRewardWrapper(gym.Wrapper if GYM_OK else object):
    """
    Adds a shaped reward for acquiring the target DIA variable.

    reward = +1.0  when target inventory variable goes from 0 → 1
    reward = +0.1  each step the variable stays acquired (sustain signal)
    reward = -0.005 per step (small survival pressure to encourage efficiency)

    Used during train_minedojo_skill.py to train each PPO skill.
    """

    def __init__(self, env: MinedojoObsWrapper, target_var_idx: int):
        if GYM_OK:
            super().__init__(env)
        else:
            self.env = env
        self.target_idx  = target_var_idx
        self._prev_val   = 0.0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        self._prev_val = float(obs["inventory"][self.target_idx])
        return obs

    def step(self, action):
        obs, ext_rew, done, info = self.env.step(action)
        curr_val = float(obs["inventory"][self.target_idx])

        shaped = -0.005  # time penalty
        if curr_val > 0.5:
            shaped += 0.1  # already acquired — small sustain bonus
        if curr_val > self._prev_val + 0.5:
            shaped += 1.0  # just acquired — big bonus
        self._prev_val = curr_val

        return obs, ext_rew + shaped, done, info


# ---------------------------------------------------------------------------
# MinedojoPPOOption — run a loaded PPO model as a DIA option
# ---------------------------------------------------------------------------

class MinedojoPPOOption(OptionPolicy):
    """
    Execute a pre-trained SB3 PPO model as a DIA skill option.

    model_path  — path to a .zip file saved by train_minedojo_skill.py
    evgs        — MinedojoEVGS instance (for subgoal evaluation)
    deterministic — use deterministic policy (True for exploitation,
                    False for exploration with temperature)

    run(env, evgs):
        env must be a MinedojoObsWrapper (has get_obs() returning dict obs).
    """

    def __init__(self, subgoal: Subgoal, cfg: OptionConfig,
                 model_or_path,
                 deterministic: bool = True):
        super().__init__(subgoal, cfg)
        self.deterministic = deterministic
        self._model = None

        if model_or_path is not None:
            self.load(model_or_path)

    def load(self, model_or_path):
        """Load model from path or accept a pre-loaded SB3 PPO instance."""
        if isinstance(model_or_path, str):
            if not SB3_OK:
                raise ImportError("stable-baselines3 is required for MinedojoPPOOption")
            self._model = SB3PPO.load(model_or_path)
        else:
            self._model = model_or_path   # already loaded

    def act(self, obs: Dict[str, np.ndarray]) -> Any:
        """Predict action for a single obs dict."""
        if self._model is None:
            # Fallback: random action (useful for skeleton testing)
            return 0
        action, _ = self._model.predict(obs, deterministic=self.deterministic)
        return action

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        """
        Execute the PPO skill until subgoal achieved or max_steps exceeded.

        env must expose get_obs() → dict obs compatible with MinedojoObsWrapper.
        """
        if not hasattr(env, "get_obs"):
            raise TypeError("MinedojoPPOOption.run requires env with get_obs()")

        obs     = env.get_obs()
        x_prev  = evgs.extract(obs)
        steps   = 0
        success = False
        trajectory: list = []

        while steps < self.cfg.max_steps:
            action   = self.act(obs)
            next_result = env.step(action)

            if len(next_result) == 5:
                next_obs, _rew, term, trunc, _info = next_result
                done = term or trunc
            else:
                next_obs, _rew, done, _info = next_result

            x_curr  = evgs.extract(obs)
            x_next  = evgs.extract(next_obs)
            succ_this = EVGS.predicate_holds(x_curr, x_next, self.subgoal)

            trajectory.append((obs, action, next_obs, succ_this))
            obs = next_obs
            steps += 1

            if succ_this:
                success = True
                if self.cfg.terminate_on_success:
                    break
            if done:
                break

        return {
            "success":   success,
            "steps":     steps,
            "trajectory": trajectory,
            "final_obs":  obs,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def load_skill_option(subgoal: Subgoal, model_path: str,
                      max_steps: int = 1000,
                      deterministic: bool = True) -> MinedojoPPOOption:
    """
    Load a pre-trained PPO skill option from disk.

    model_path — e.g. "models/minedojo/skill_wood.zip"
    """
    cfg = OptionConfig(max_steps=max_steps, terminate_on_success=True)
    return MinedojoPPOOption(subgoal, cfg, model_path, deterministic=deterministic)
