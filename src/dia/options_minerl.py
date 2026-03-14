# src/dia/options_minerl.py
"""
PPO-based skill options for MineRL 3D Minecraft.

Each skill (PPO model) is trained via scripts/train_minerl_skill.py,
then loaded here for execution during Phase 2 transfer.

MineRL obs:   obs['pov'] (H,W,3), obs['inventory'] (flat dict)
MineRL action: Dict space with keys attack/forward/jump/camera/use/…

Architecture
------------
MineRLObsWrapper
  - Converts MineRL obs → {'rgb': (64,64,3), 'inventory': (9,)} for SB3

MineRLFlatActionWrapper
  - Flattens the Dict action space to MultiDiscrete(15) for SB3 PPO

MineRLPPOOption
  - Wraps a pre-trained SB3 PPO model; same .run(env, evgs) interface

RandomMineRLOption
  - Uses random actions; serves as a placeholder when no PPO model is trained
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
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

try:
    from stable_baselines3 import PPO as SB3PPO
    SB3_OK = True
except Exception:
    SB3PPO = None; SB3_OK = False


# ── Observation constants ─────────────────────────────────────────────────────
OBS_H   = 64    # RGB height fed to CNN
OBS_W   = 64    # RGB width  fed to CNN
INV_DIM = 9     # DIA variable vector length

# ── Discrete action set for MineRL ────────────────────────────────────────────
# Maps index → (action_key_changes, camera_delta)
# camera: [pitch_delta, yaw_delta] in degrees
_NOOP: Dict[str, Any] = {
    "attack": 0, "back": 0, "forward": 0, "jump": 0,
    "left": 0, "right": 0, "sneak": 0, "sprint": 0, "use": 0,
    "camera": np.array([0.0, 0.0]),
    "drop": 0, "inventory": 0, "pickItem": 0, "swapHands": 0,
    "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0,
    "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0,
    "hotbar.9": 0, "ESC": 0,
}

# Each entry: list of (key, value) overrides on top of _NOOP
_ACTION_DEFS = [
    [],                                          # 0: noop
    [("forward", 1)],                            # 1: forward
    [("back", 1)],                               # 2: back
    [("left", 1)],                               # 3: strafe left
    [("right", 1)],                              # 4: strafe right
    [("jump", 1)],                               # 5: jump
    [("attack", 1)],                             # 6: attack (mine/chop)
    [("use", 1)],                                # 7: use (place/craft/open)
    [("forward", 1), ("attack", 1)],             # 8: forward + attack
    [("forward", 1), ("jump", 1)],               # 9: forward + jump
    [("forward", 1), ("sprint", 1)],             # 10: sprint forward
    [("camera", np.array([-15.0, 0.0]))],        # 11: camera up
    [("camera", np.array([15.0, 0.0]))],         # 12: camera down
    [("camera", np.array([0.0, -15.0]))],        # 13: camera left
    [("camera", np.array([0.0, 15.0]))],         # 14: camera right
]
N_ACTIONS = len(_ACTION_DEFS)


def idx_to_minerl_action(idx: int) -> Dict[str, Any]:
    """Convert a discrete action index to a MineRL-compatible action dict."""
    act = dict(_NOOP)
    act["camera"] = np.array([0.0, 0.0])
    for key, val in _ACTION_DEFS[int(idx) % N_ACTIONS]:
        act[key] = val
    return act


# ---------------------------------------------------------------------------
# MineRLObsWrapper — convert raw MineRL obs to SB3-compatible dict
# ---------------------------------------------------------------------------

class MineRLObsWrapper(gym.Wrapper if GYM_OK else object):
    """
    Wraps a MineRL environment.

    Input:  raw MineRL obs dict with 'pov' (image) + 'inventory' (flat dict)
    Output: {'rgb': (64,64,3) uint8, 'inventory': (9,) float32}
    """

    def __init__(self, env, evgs: EVGS):
        if GYM_OK:
            super().__init__(env)
        else:
            self.env = env
        self.evgs = evgs
        self._last_raw: Any = None

        if GYM_OK and spaces is not None:
            self.observation_space = spaces.Dict({
                "rgb":       spaces.Box(0, 255, (OBS_H, OBS_W, 3), dtype=np.uint8),
                "inventory": spaces.Box(0.0, 1.0, (INV_DIM,), dtype=np.float32),
            })
            self.action_space = spaces.Discrete(N_ACTIONS)

    def _convert(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        rgb = self._resize_pov(raw_obs)
        inv = self.evgs.extract(raw_obs).astype(np.float32)
        return {"rgb": rgb, "inventory": inv}

    @staticmethod
    def _resize_pov(obs: Any) -> np.ndarray:
        """Resize obs['pov'] to (OBS_H, OBS_W, 3)."""
        try:
            img = np.asarray(obs["pov"], dtype=np.uint8)
            if img.ndim == 3:
                if CV2_OK:
                    return cv2.resize(img, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)
                # Fallback: crop center
                h, w = img.shape[:2]
                cy, cx = h // 2, w // 2
                half_h, half_w = OBS_H // 2, OBS_W // 2
                cropped = img[cy-half_h:cy+half_h, cx-half_w:cx+half_w]
                if cropped.shape[:2] == (OBS_H, OBS_W):
                    return cropped
        except Exception:
            pass
        return np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8)

    def reset(self, **kwargs):
        raw = self.env.reset(**kwargs)
        # gym 0.23 returns just obs (not (obs, info))
        if isinstance(raw, tuple):
            raw = raw[0]
        self._last_raw = raw
        return self._convert(raw)

    def step(self, action):
        # Convert discrete index to MineRL dict action
        minerl_action = idx_to_minerl_action(int(action))
        result = self.env.step(minerl_action)
        if len(result) == 5:
            raw, rew, term, trunc, info = result
            done = term or trunc
        else:
            raw, rew, done, info = result
        self._last_raw = raw
        return self._convert(raw), float(rew), bool(done), info

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Return current obs in wrapped format (for OptionPolicy compatibility)."""
        if self._last_raw is not None:
            return self._convert(self._last_raw)
        return {"rgb": np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8),
                "inventory": np.zeros(INV_DIM, dtype=np.float32)}

    def get_raw_obs(self) -> Any:
        return self._last_raw


# ---------------------------------------------------------------------------
# ItemRewardWrapper — shape reward for individual skill training
# ---------------------------------------------------------------------------

class ItemRewardWrapper(gym.Wrapper if GYM_OK else object):
    """
    Adds a shaped reward for acquiring the target DIA variable.

    reward = +1.0  when target inventory variable goes 0 → 1
    reward = +0.1  each step the variable stays acquired
    reward = -0.005 per step (time pressure)
    """

    def __init__(self, env: MineRLObsWrapper, target_var_idx: int):
        if GYM_OK:
            super().__init__(env)
        else:
            self.env = env
        self.target_idx = target_var_idx
        self._prev_val  = 0.0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        self._prev_val = float(obs["inventory"][self.target_idx])
        return obs

    def step(self, action):
        obs, ext_rew, done, info = self.env.step(action)
        curr_val = float(obs["inventory"][self.target_idx])

        shaped = -0.005
        if curr_val > 0.5:
            shaped += 0.1
        if curr_val > self._prev_val + 0.5:
            shaped += 1.0
        self._prev_val = curr_val
        return obs, ext_rew + shaped, done, info


# ---------------------------------------------------------------------------
# MineRLPPOOption — run a loaded SB3 PPO model as a DIA skill option
# ---------------------------------------------------------------------------

class MineRLPPOOption(OptionPolicy):
    """
    Execute a pre-trained SB3 PPO model as a DIA skill option in MineRL.

    model_path — path to a .zip file saved by train_minerl_skill.py
    env must be a MineRLObsWrapper (exposes .get_obs() and .step(int_idx))
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
        if isinstance(model_or_path, str):
            if not SB3_OK:
                raise ImportError("stable-baselines3 required for MineRLPPOOption")
            self._model = SB3PPO.load(model_or_path)
        else:
            self._model = model_or_path

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        if self._model is None:
            return int(np.random.randint(N_ACTIONS))
        action, _ = self._model.predict(obs, deterministic=self.deterministic)
        return int(action)

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        if not hasattr(env, "get_obs"):
            raise TypeError("MineRLPPOOption.run requires env with get_obs()")
        obs     = env.get_obs()
        steps   = 0
        success = False
        trajectory: list = []

        while steps < self.cfg.max_steps:
            action      = self.act(obs)
            next_obs, _rew, done, _info = env.step(action)

            x_curr = evgs.extract(obs)
            x_next = evgs.extract(next_obs)
            succ   = EVGS.predicate_holds(x_curr, x_next, self.subgoal)
            trajectory.append((obs, action, next_obs, succ))
            obs     = next_obs
            steps  += 1

            if succ:
                success = True
                if self.cfg.terminate_on_success:
                    break
            if done:
                break

        return {"success": success, "steps": steps,
                "trajectory": trajectory, "final_obs": obs}


# ---------------------------------------------------------------------------
# RandomMineRLOption — placeholder when no PPO model is available
# ---------------------------------------------------------------------------

class RandomMineRLOption(OptionPolicy):
    """
    Runs random actions as a placeholder skill.
    Useful for testing the transfer pipeline before PPO training.
    """

    def __init__(self, subgoal: Subgoal, cfg: OptionConfig):
        super().__init__(subgoal, cfg)

    def act(self, obs) -> int:
        return int(np.random.randint(N_ACTIONS))

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        obs    = env.get_obs() if hasattr(env, "get_obs") else {}
        steps  = 0
        success = False
        trajectory: list = []

        while steps < self.cfg.max_steps:
            action = self.act(obs)
            next_obs, _rew, done, _info = env.step(action)

            x_curr = evgs.extract(obs)
            x_next = evgs.extract(next_obs)
            succ   = EVGS.predicate_holds(x_curr, x_next, self.subgoal)
            trajectory.append((obs, action, next_obs, succ))
            obs    = next_obs
            steps += 1

            if succ:
                success = True
                if self.cfg.terminate_on_success:
                    break
            if done:
                break

        return {"success": success, "steps": steps,
                "trajectory": trajectory, "final_obs": obs}


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def load_skill_option(subgoal: Subgoal, model_path: str,
                      max_steps: int = 2000,
                      deterministic: bool = True) -> MineRLPPOOption:
    """Load a pre-trained PPO skill from disk."""
    cfg = OptionConfig(max_steps=max_steps, terminate_on_success=True)
    return MineRLPPOOption(subgoal, cfg, model_path, deterministic=deterministic)


def make_random_option(subgoal: Subgoal, max_steps: int = 500) -> RandomMineRLOption:
    """Create a random-action placeholder option."""
    cfg = OptionConfig(max_steps=max_steps, terminate_on_success=False)
    return RandomMineRLOption(subgoal, cfg)
