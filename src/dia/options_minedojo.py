# src/dia/options_minedojo.py
"""
PPO-based skill options for MineDojo 3D Minecraft.

Each skill (PPO model) is trained separately via scripts/train_minedojo_skill.py,
then loaded here for execution during Phase 2 transfer.

Architecture
------------
MinedojoPPOOption
  - Wraps a pre-trained SB3 PPO model
  - Preprocesses MineDojo's multimodal obs → compact (image + inventory + clip_goal) feature
  - Runs until subgoal achieved (var increases) or max_steps
  - Same .run(env, evgs) interface as CoinRun PixelStackPPOOption

MinedojoObsWrapper  (gym.Wrapper)
  - Converts raw MineDojo obs dict → compact obs for SB3 training
  - Outputs Dict space: {"rgb": Box(H,W,3), "inventory": Box(9,), ["clip_goal": Box(512,)]}
  - clip_goal is a CLIP text embedding injected from outside (set via set_goal_embedding())
  - Used during training AND inference

ItemRewardWrapper  (gym.RewardWrapper)
  - Adds +1 reward when target inventory item count increases
  - Optionally adds CLIP cosine-similarity dense reward (clip_weight * cos_sim)
  - Used during PPO skill training

Observation design for SB3
---------------------------
  "rgb":       (64, 64, 3) uint8  — downsampled from MineDojo's 160×256
  "inventory": (9,)       float32 — DIA variable vector from evgs_minedojo
  "clip_goal": (512,)     float32 — CLIP text embedding (optional, only when use_clip=True)

SB3 MultiInputPolicy handles this Dict space with:
  - CnnPolicy branch for "rgb"
  - MlpPolicy branch for "inventory"
  - MlpPolicy branch for "clip_goal" (if present)
  All branches are concatenated before the action head.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

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
OBS_H   = 64    # RGB height fed to CNN
OBS_W   = 64    # RGB width  fed to CNN
INV_DIM = 9     # DIA variable vector length
GOAL_DIM = 512  # CLIP ViT-B/32 text embedding dimension


# ---------------------------------------------------------------------------
# MinedojoObsWrapper — convert raw MineDojo obs to SB3-compatible dict
# ---------------------------------------------------------------------------

class MinedojoObsWrapper(gym.Wrapper if GYM_OK else object):
    """
    Wraps a MineDojo environment.

    Input:  raw MineDojo obs dict with 'rgb', 'inventory', ...
    Output: {'rgb': (64,64,3) uint8, 'inventory': (9,) float32}
            + optionally 'clip_goal': (512,) float32 when goal_embedding is set.

    Also exposes get_obs() and set_goal_embedding() for inference-time
    VLM→CLIP goal injection.
    """

    def __init__(self, env, evgs: EVGS,
                 goal_embedding: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        env             : MineDojo environment
        evgs            : EVGS instance for inventory extraction
        goal_embedding  : [512] float32 CLIP text embedding (optional).
                          If provided, 'clip_goal' is added to the obs space.
                          Can be updated at any time via set_goal_embedding().
        """
        if GYM_OK:
            super().__init__(env)
        else:
            self.env = env
        self.evgs = evgs
        self._last_obs: Any = None
        self._goal_embedding: Optional[np.ndarray] = (
            goal_embedding.astype(np.float32) if goal_embedding is not None else None
        )

        if GYM_OK and spaces is not None:
            obs_spaces: Dict[str, Any] = {
                "rgb":       spaces.Box(0, 255, (OBS_H, OBS_W, 3), dtype=np.uint8),
                "inventory": spaces.Box(0.0, 1.0, (INV_DIM,), dtype=np.float32),
            }
            if goal_embedding is not None:
                obs_spaces["clip_goal"] = spaces.Box(
                    -1.0, 1.0, (GOAL_DIM,), dtype=np.float32
                )
            self.observation_space = spaces.Dict(obs_spaces)

    def set_goal_embedding(self, embed: np.ndarray) -> None:
        """
        Update the CLIP goal embedding injected into every obs.

        Called at inference time by the transfer script after the VLM
        produces a new instruction for the current skill.
        """
        self._goal_embedding = embed.astype(np.float32)

    def _convert(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        rgb = self._get_rgb(raw_obs)
        inv = self.evgs.extract(raw_obs).astype(np.float32)
        out: Dict[str, np.ndarray] = {"rgb": rgb, "inventory": inv}
        if self._goal_embedding is not None:
            out["clip_goal"] = self._goal_embedding
        return out

    @staticmethod
    def _get_rgb(obs: Any) -> np.ndarray:
        """Extract and resize RGB frame from MineDojo obs."""
        try:
            import cv2
            img = np.asarray(obs["rgb"], dtype=np.uint8)
            # MineDojo returns (3, H, W) channel-first — transpose to (H, W, 3)
            if img.ndim == 3 and img.shape[0] == 3 and img.shape[0] != img.shape[2]:
                img = img.transpose(1, 2, 0)
            if img.ndim == 3 and img.shape[2] == 3:
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

    # Lookup table: DIA 25-action index → MineDojo 8-dim MultiDiscrete
    # Dims: [forward/back, left/right, jump/sneak/sprint, pitch_bin, yaw_bin, fn, craft_arg, slot]
    # Noop bins for pitch/yaw: (25-1)//2 = 12  (15°/bin, 12*15-180=0°)
    _DIA_TO_MD = np.array([
        [0, 0, 0, 12, 12, 0, 0, 0],  #  0: noop
        [1, 0, 0, 12, 12, 0, 0, 0],  #  1: forward
        [2, 0, 0, 12, 12, 0, 0, 0],  #  2: back
        [0, 1, 0, 12, 12, 0, 0, 0],  #  3: left
        [0, 2, 0, 12, 12, 0, 0, 0],  #  4: right
        [0, 0, 1, 12, 12, 0, 0, 0],  #  5: jump
        [0, 0, 0, 12, 12, 3, 0, 0],  #  6: attack
        [0, 0, 0, 12, 12, 1, 0, 0],  #  7: use
        [1, 0, 0, 12, 12, 3, 0, 0],  #  8: forward+attack
        [1, 0, 1, 12, 12, 0, 0, 0],  #  9: forward+jump
        [1, 0, 3, 12, 12, 0, 0, 0],  # 10: forward+sprint
        [0, 0, 0, 11, 12, 0, 0, 0],  # 11: camera pitch -15°
        [0, 0, 0, 13, 12, 0, 0, 0],  # 12: camera pitch +15°
        [0, 0, 0, 12, 11, 0, 0, 0],  # 13: camera yaw -15°
        [0, 0, 0, 12, 13, 0, 0, 0],  # 14: camera yaw +15°
        [1, 0, 3, 12, 12, 3, 0, 0],  # 15: forward+sprint+attack
        [0, 0, 0, 11, 12, 0, 0, 0],  # 16: camera pitch -5° (approx as -15°)
        [0, 0, 0, 13, 12, 0, 0, 0],  # 17: camera pitch +5° (approx as +15°)
        [0, 0, 0, 12, 11, 0, 0, 0],  # 18: camera yaw  -5° (approx as -15°)
        [0, 0, 0, 12, 13, 0, 0, 0],  # 19: camera yaw  +5° (approx as +15°)
        [0, 0, 0, 12, 12, 5, 0, 0],  # 20: equip slot 0
        [0, 0, 0, 12, 12, 5, 0, 1],  # 21: equip slot 1
        [1, 0, 0, 12, 12, 1, 0, 0],  # 22: forward+use
        [0, 0, 1, 12, 12, 3, 0, 0],  # 23: jump+attack
        [1, 0, 1, 12, 12, 3, 0, 0],  # 24: forward+jump+attack
    ], dtype=np.int64)

    def step(self, action):
        # Convert DIA integer action index → MineDojo 8-dim MultiDiscrete array
        if isinstance(action, (int, np.integer)):
            action = self._DIA_TO_MD[int(action) % 25]
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
        out: Dict[str, np.ndarray] = {
            "rgb":       np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8),
            "inventory": np.zeros(INV_DIM, dtype=np.float32),
        }
        if self._goal_embedding is not None:
            out["clip_goal"] = self._goal_embedding
        return out

    def get_raw_obs(self) -> Any:
        """Return the most recent raw (unwrapped) MineDojo observation."""
        return self._last_obs

    def step_command(self, action_type: str, item_name: str):
        """
        Execute a high-level named command (craft / smelt / equip) by
        converting it to a discrete action index and calling step().

        action_type — 'craft', 'smelt', 'equip', or 'use'
        item_name   — Minecraft item string (e.g. 'stone_pickaxe')

        Falls back to the use action (index 7) for any unrecognised command.
        Returns (obs_dict, reward, done, info) — same as step().
        """
        # For crafting/smelting: repeated use-interactions are the best we can
        # do without direct GUI control.  We cycle the furnace/crafting macro.
        if action_type in ("craft", "smelt", "use"):
            discrete_idx = 7  # use (right-click / interact)
        elif action_type == "equip":
            discrete_idx = 20  # hotbar.1 — grab first slot
        else:
            discrete_idx = 7
        return self.step(discrete_idx)

    @property
    def action_space(self):
        return self.env.action_space


# Public alias matching the spec naming convention (MineDojoObsWrapper)
MineDojoObsWrapper = MinedojoObsWrapper


# ---------------------------------------------------------------------------
# ItemRewardWrapper — shape reward for skill training
# ---------------------------------------------------------------------------

class ItemRewardWrapper(gym.Wrapper if GYM_OK else object):
    """
    Adds a shaped reward for acquiring the target DIA variable.

    reward = +1.0       when target inventory variable goes from 0 → 1
    reward = +0.1       each step the variable stays acquired (sustain signal)
    reward = -0.005     per step (small survival pressure)
    reward += clip_weight * cos_sim(frame, goal_embed)   (if clip_encoder provided)

    The CLIP dense reward provides a gradient signal toward the visual goal
    before the item reward fires, making sparse skill acquisition much easier
    to learn.

    Used during train_minedojo_skill.py to train each PPO skill.
    """

    def __init__(self, env: MinedojoObsWrapper, target_var_idx: int,
                 clip_encoder=None, clip_weight: float = 0.1):
        """
        Parameters
        ----------
        env             : MinedojoObsWrapper
        target_var_idx  : index into the inventory vector for the target item
        clip_encoder    : CLIPGoalEncoder instance (optional).
                          If provided, CLIP cosine-similarity is added to reward.
                          The goal embedding is read from env._goal_embedding.
        clip_weight     : weight for CLIP reward term (default 0.1)
        """
        if GYM_OK:
            super().__init__(env)
        else:
            self.env = env
        self.target_idx   = target_var_idx
        self._prev_val    = 0.0
        self._clip_enc    = clip_encoder
        self._clip_weight = clip_weight

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
            shaped += 0.1   # already acquired — small sustain bonus
        if curr_val > self._prev_val + 0.5:
            shaped += 1.0   # just acquired — big bonus
        self._prev_val = curr_val

        # ── CLIP dense reward ─────────────────────────────────────────────
        if (self._clip_enc is not None
                and "clip_goal" in obs
                and "rgb" in obs):
            clip_r = self._clip_enc.clip_reward(obs["rgb"], obs["clip_goal"])
            shaped += self._clip_weight * clip_r

        return obs, ext_rew + shaped, done, info


# ---------------------------------------------------------------------------
# MinedojoPPOOption — run a loaded PPO model as a DIA option
# ---------------------------------------------------------------------------

class MinedojoPPOOption(OptionPolicy):
    """
    Execute a pre-trained SB3 PPO model as a DIA skill option.

    model_path    — path to a .zip file saved by train_minedojo_skill.py
    deterministic — use deterministic policy (True for exploitation)

    set_directive(actions) — prepend VLM-supplied raw actions before PPO takes over
    set_goal_embedding(embed) — update the CLIP goal injected into the env's obs

    run(env, evgs):
        env must be a MinedojoObsWrapper (has get_obs() and set_goal_embedding()).
    """

    def __init__(self, subgoal: Subgoal, cfg: OptionConfig,
                 model_or_path,
                 deterministic: bool = True):
        super().__init__(subgoal, cfg)
        self.deterministic = deterministic
        self._model = None
        self._directive: List[int] = []
        self._goal_embedding: Optional[np.ndarray] = None

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

    def set_directive(self, actions: List[int]) -> None:
        """
        Prepend a VLM-supplied action sequence to execute before the PPO policy.
        Called by the transfer script after the VLM suggests an approach direction.
        """
        self._directive = list(actions)

    def set_goal_embedding(self, embed: np.ndarray) -> None:
        """
        Store a CLIP goal embedding that will be injected into the env's obs
        at the start of run().  Called by the transfer script after the VLM
        produces an instruction for the current skill.
        """
        self._goal_embedding = embed.astype(np.float32)

    def act(self, obs: Dict[str, np.ndarray]) -> Any:
        """Predict action for a single obs dict (pops directive first)."""
        if self._directive:
            return int(self._directive.pop(0))
        if self._model is None:
            return 0  # fallback: noop
        action, _ = self._model.predict(obs, deterministic=self.deterministic)
        return action

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        """
        Execute the PPO skill until subgoal achieved or max_steps exceeded.

        env must expose get_obs() → dict obs compatible with MinedojoObsWrapper.
        If a goal_embedding is set, it is pushed to env.set_goal_embedding()
        before the loop starts so that every step's obs includes 'clip_goal'.
        """
        if not hasattr(env, "get_obs"):
            raise TypeError("MinedojoPPOOption.run requires env with get_obs()")

        # Push VLM-derived CLIP goal into the env wrapper before stepping
        if self._goal_embedding is not None and hasattr(env, "set_goal_embedding"):
            env.set_goal_embedding(self._goal_embedding)

        obs     = env.get_obs()
        steps   = 0
        success = False
        trajectory: list = []

        while steps < self.cfg.max_steps:
            action      = self.act(obs)
            next_result = env.step(action)

            if len(next_result) == 5:
                next_obs, _rew, term, trunc, _info = next_result
                done = term or trunc
            else:
                next_obs, _rew, done, _info = next_result

            x_curr    = evgs.extract(obs)
            x_next    = evgs.extract(next_obs)
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
            "success":    success,
            "steps":      steps,
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


# ---------------------------------------------------------------------------
# Discrete action index → MineDojo action dict translation
# ---------------------------------------------------------------------------
# 25-action set matching options_minerl._ACTION_DEFS.
# MineDojo action dict uses the same keys but also has craft/smelt/place/equip.
# We use env.action_space.no_op() as a base; only movement keys are toggled here.
# Craft/smelt/place/equip are handled via MineDojoObsWrapper.step_command().

_N_ACTIONS = 25

def _idx_to_minedojo_action(env_no_op: dict, idx: int) -> dict:
    """
    Translate a discrete action index (0-24) to a MineDojo action dict.

    env_no_op — the no-op action from env.action_space.no_op()
    Returns a *copy* of no_op with the appropriate fields overridden.
    """
    import copy
    act = copy.deepcopy(env_no_op)

    idx = int(idx) % _N_ACTIONS

    if idx == 0:
        pass  # noop
    elif idx == 1:
        act["forward"] = 1
    elif idx == 2:
        act["back"] = 1
    elif idx == 3:
        act["left"] = 1
    elif idx == 4:
        act["right"] = 1
    elif idx == 5:
        act["jump"] = 1
    elif idx == 6:
        act["attack"] = 1
    elif idx == 7:
        act["use"] = 1
    elif idx == 8:
        act["forward"] = 1; act["attack"] = 1
    elif idx == 9:
        act["forward"] = 1; act["jump"] = 1
    elif idx == 10:
        act["forward"] = 1; act["sprint"] = 1
    elif idx == 11:
        act["camera"] = np.array([-15.0, 0.0])
    elif idx == 12:
        act["camera"] = np.array([15.0, 0.0])
    elif idx == 13:
        act["camera"] = np.array([0.0, -15.0])
    elif idx == 14:
        act["camera"] = np.array([0.0, 15.0])
    elif idx == 15:
        act["forward"] = 1; act["sprint"] = 1; act["attack"] = 1
    elif idx == 16:
        act["camera"] = np.array([-5.0, 0.0])
    elif idx == 17:
        act["camera"] = np.array([5.0, 0.0])
    elif idx == 18:
        act["camera"] = np.array([0.0, -5.0])
    elif idx == 19:
        act["camera"] = np.array([0.0, 5.0])
    elif idx in (20, 21):
        pass  # hotbar not needed; equip handled by step_command
    elif idx == 22:
        act["forward"] = 1; act["use"] = 1
    elif idx == 23:
        act["jump"] = 1; act["attack"] = 1
    elif idx == 24:
        act["forward"] = 1; act["jump"] = 1; act["attack"] = 1

    return act


# ---------------------------------------------------------------------------
# MineDojoObsWrapper — primary env wrapper for MineDojo with step_command API
# ---------------------------------------------------------------------------

try:
    import cv2 as _cv2
    _CV2_OK = True
except Exception:
    _cv2 = None; _CV2_OK = False


class MineDojoObsWrapper:
    """
    Wraps a MineDojo environment providing a common interface for DIA skill execution.

    Key differences from MineDojo raw env:
    - MineDojo rgb is (3, H, W) channel-first; we convert to (H, W, 3).
    - step(discrete_idx) translates our 25-action space to MineDojo action dict.
    - step_command(action_type, item_name) sends craft/smelt/place/equip commands.

    Outputs:
      reset() / step() return {'rgb': (64,64,3) uint8, 'inventory': (9,) float32}

    Compatible with MinedojoPPOOption.run(env, evgs) interface.
    """

    def __init__(self, env, evgs: EVGS,
                 goal_embedding: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        env            : raw MineDojo environment
        evgs           : EVGS instance (evgs_minedojo.make_minedojo_evgs())
        goal_embedding : optional (512,) CLIP embedding; adds 'clip_goal' to obs
        """
        self.env = env
        self.evgs = evgs
        self._last_raw: Any = None
        self._goal_embedding: Optional[np.ndarray] = (
            goal_embedding.astype(np.float32) if goal_embedding is not None else None
        )
        # Cache the no-op action for action translation
        self._no_op_action: Optional[dict] = None

    # -- Goal embedding --

    def set_goal_embedding(self, embed: np.ndarray) -> None:
        """Update the CLIP goal embedding injected into every obs."""
        self._goal_embedding = embed.astype(np.float32)

    # -- Core interface --

    def reset(self, **kwargs) -> Dict[str, np.ndarray]:
        """Reset env and return wrapped obs dict."""
        raw = self.env.reset(**kwargs)
        raw = raw[0] if isinstance(raw, tuple) else raw
        self._last_raw = raw
        self._no_op_action = None  # invalidate cached no-op
        return self._convert(raw)

    def step(self, discrete_idx: int):
        """
        Translate discrete_idx (0-24) → MineDojo action dict and step env.

        Returns (wrapped_obs, reward, done, info).
        """
        no_op = self._get_no_op()
        action = _idx_to_minedojo_action(no_op, discrete_idx)
        return self._step_raw(action)

    def step_command(self, action_type: str, item_name: str):
        """
        Send a craft/smelt/place/equip command directly to MineDojo.

        action_type : one of 'craft', 'craft_with_table', 'smelt', 'place', 'equip'
        item_name   : Minecraft item id, e.g. 'stone_pickaxe', 'furnace', 'iron_ingot'

        Returns (wrapped_obs, reward, done, info).
        """
        no_op = self._get_no_op()
        import copy
        act = copy.deepcopy(no_op)

        # MineDojo supports these action keys directly in the action dict.
        # Each key maps to an item name string (the no-op value is '' or 'none').
        if action_type in ("craft", "craft_with_table", "smelt", "place", "equip"):
            # Some MineDojo versions use 'craft' for table crafting and
            # 'craft_with_table' for 3x3 recipe crafting.  We set whichever key
            # exists in the action space.
            if action_type in act:
                act[action_type] = item_name
            elif action_type == "craft_with_table" and "craft" in act:
                # Fallback: use 'craft' key if 'craft_with_table' not present
                act["craft"] = item_name
            elif action_type == "craft" and "craft_with_table" in act:
                # Other direction: prefer table crafting key
                act["craft_with_table"] = item_name
        else:
            raise ValueError(f"Unknown action_type: {action_type!r}. "
                             f"Use 'craft', 'craft_with_table', 'smelt', 'place', or 'equip'.")

        return self._step_raw(act)

    # -- Obs access --

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Return current wrapped obs (for compatibility with OptionPolicy.run)."""
        if self._last_raw is not None:
            return self._convert(self._last_raw)
        out: Dict[str, np.ndarray] = {
            "rgb":       np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8),
            "inventory": np.zeros(INV_DIM, dtype=np.float32),
        }
        if self._goal_embedding is not None:
            out["clip_goal"] = self._goal_embedding
        return out

    def get_raw_obs(self) -> Any:
        """Return the last raw MineDojo observation."""
        return self._last_raw

    # -- Internal helpers --

    def _convert(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        """Convert raw MineDojo obs → wrapped obs dict."""
        rgb = self._get_rgb(raw_obs)
        inv = self.evgs.extract(raw_obs).astype(np.float32)
        out: Dict[str, np.ndarray] = {"rgb": rgb, "inventory": inv}
        if self._goal_embedding is not None:
            out["clip_goal"] = self._goal_embedding
        return out

    def _get_rgb(self, raw_obs: Any) -> np.ndarray:
        """
        Extract RGB frame from MineDojo obs and convert to (H, W, 3) uint8.

        MineDojo returns rgb as (3, H, W) channel-first.
        We transpose to (H, W, 3) then resize to (OBS_H, OBS_W, 3).
        """
        try:
            img = np.asarray(raw_obs["rgb"], dtype=np.uint8)
            # Handle channel-first format (3, H, W) → (H, W, 3)
            if img.ndim == 3 and img.shape[0] == 3 and img.shape[0] != img.shape[2]:
                img = img.transpose(1, 2, 0)
            if img.ndim == 3 and img.shape[2] == 3:
                if _CV2_OK:
                    return _cv2.resize(img, (OBS_W, OBS_H),
                                       interpolation=_cv2.INTER_AREA)
                # Fallback: center-crop
                h, w = img.shape[:2]
                cy, cx = h // 2, w // 2
                hh, hw = OBS_H // 2, OBS_W // 2
                cropped = img[cy - hh:cy + hh, cx - hw:cx + hw]
                if cropped.shape[:2] == (OBS_H, OBS_W):
                    return cropped
        except Exception:
            pass
        return np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8)

    def _get_inv_counts(self, raw_obs: Any) -> Dict[str, int]:
        """
        Return a flat {item_name: count} dict from a MineDojo raw observation.

        Useful for prerequisite checking without running the full EVGS extraction.
        """
        result: Dict[str, int] = {}
        try:
            inv = raw_obs.get("inventory", {}) if isinstance(raw_obs, dict) else {}
            if isinstance(inv, dict):
                names = inv.get("name", [])
                qtys  = inv.get("quantity", [])
            elif hasattr(inv, "dtype") and hasattr(inv.dtype, "names"):
                names = inv["name"] if "name" in inv.dtype.names else []
                qtys  = inv["quantity"] if "quantity" in inv.dtype.names else []
            else:
                return result
            for name, qty in zip(names, qtys):
                n = str(name).lower().strip().strip("'\"").rstrip("\x00")
                if n and n not in ("air", "none", ""):
                    result[n] = result.get(n, 0) + int(qty)
        except Exception:
            pass
        return result

    def _step_raw(self, action: dict):
        """Step the raw MineDojo env with action dict; return wrapped tuple."""
        result = self.env.step(action)
        if len(result) == 5:
            raw, rew, term, trunc, info = result
            done = term or trunc
        else:
            raw, rew, done, info = result
        self._last_raw = raw
        return self._convert(raw), float(rew), bool(done), info

    def _get_no_op(self) -> dict:
        """Return (and cache) the no-op action dict from the env's action space."""
        if self._no_op_action is None:
            try:
                self._no_op_action = self.env.action_space.no_op()
            except Exception:
                # Fallback minimal no-op if env is not fully initialized
                self._no_op_action = {
                    "forward": 0, "back": 0, "left": 0, "right": 0,
                    "jump": 0, "sneak": 0, "sprint": 0,
                    "use": 0, "attack": 0, "drop": 0,
                    "camera": np.array([0.0, 0.0]),
                }
        return self._no_op_action

    # -- Passthrough attrs --

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space


# ---------------------------------------------------------------------------
# InventoryConditionedCraftOption — deterministic craft/smelt option
# ---------------------------------------------------------------------------

# Skills handled by this deterministic option (no BC policy needed)
CRAFT_SKILLS = frozenset({"furnace", "stonepickaxe", "iron", "ironpickaxe"})

# Required inventory items for each craft skill (item_name, min_count)
_CRAFT_PREREQS: Dict[str, List[tuple]] = {
    "stonepickaxe": [("cobblestone", 3)],  # wood is obtained along the way
    "furnace":      [("cobblestone", 8)],
    "iron":         [("iron_ore", 1), ("coal", 1)],
    "ironpickaxe":  [("iron_ingot", 3)],
}

# Craft step sequences per skill: list of (action_type, item_name) tuples
_CRAFT_SEQUENCES: Dict[str, List[tuple]] = {
    "stonepickaxe": [
        ("craft",            "planks"),           # log → 4 planks (2x2, no table)
        ("craft",            "sticks"),            # 2 planks → 4 sticks (2x2)
        ("craft",            "crafting_table"),    # 4 planks → 1 table (2x2)
        ("place",            "crafting_table"),    # place table in front
        ("craft_with_table", "stone_pickaxe"),     # cobblestone + sticks (3x3)
        ("equip",            "stone_pickaxe"),     # equip for mining
    ],
    "furnace": [
        ("craft",            "planks"),
        ("craft",            "crafting_table"),
        ("place",            "crafting_table"),
        ("craft_with_table", "furnace"),           # 8 cobblestone (3x3)
        ("place",            "furnace"),           # place furnace
    ],
    "iron": [
        ("smelt",            "iron_ingot"),        # iron_ore + coal → iron_ingot
    ],
    "ironpickaxe": [
        ("craft",            "planks"),
        ("craft",            "sticks"),
        ("craft",            "crafting_table"),
        ("place",            "crafting_table"),
        ("craft_with_table", "iron_pickaxe"),      # iron_ingot + sticks (3x3)
        ("equip",            "iron_pickaxe"),
    ],
}


def _parse_inv_counts(raw_obs: Any) -> Dict[str, int]:
    """Return {item_name: count} from a MineDojo raw observation dict."""
    result: Dict[str, int] = {}
    try:
        inv = raw_obs.get("inventory", {}) if isinstance(raw_obs, dict) else {}
        if isinstance(inv, dict):
            names = inv.get("name", [])
            qtys  = inv.get("quantity", [])
        elif hasattr(inv, "dtype") and hasattr(inv.dtype, "names"):
            names = inv["name"] if "name" in inv.dtype.names else []
            qtys  = inv["quantity"] if "quantity" in inv.dtype.names else []
        else:
            return result
        for name, qty in zip(names, qtys):
            n = str(name).lower().strip().strip("'\"").rstrip("\x00")
            if n and n not in ("air", "none", ""):
                result[n] = result.get(n, 0) + int(qty)
    except Exception:
        pass
    return result


class InventoryConditionedCraftOption(OptionPolicy):
    """
    Deterministic craft/smelt option for MineDojo.

    Checks prerequisites from inventory, then issues craft/smelt/place/equip
    commands via env.step_command().  No BC policy is needed because these
    operations are deterministic once prerequisites are met.

    Compatible with the OptionPolicy.run(env, evgs) interface.
    env must be a MineDojoObsWrapper (exposes step_command and get_obs).
    """

    def __init__(self, subgoal: Subgoal, cfg: OptionConfig, skill_name: str):
        """
        Parameters
        ----------
        subgoal     : Subgoal(var_index=..., predicate=UP) for the target variable
        cfg         : OptionConfig (max_steps used as overall timeout)
        skill_name  : one of 'furnace', 'stonepickaxe', 'iron', 'ironpickaxe'
        """
        super().__init__(subgoal, cfg)
        self.skill_name = skill_name
        if skill_name not in CRAFT_SKILLS:
            raise ValueError(
                f"InventoryConditionedCraftOption: skill_name must be one of "
                f"{sorted(CRAFT_SKILLS)}, got {skill_name!r}"
            )

    def act(self, obs) -> Any:
        """Not used; this option executes via run() directly."""
        return 0

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        """
        Execute craft sequence if prerequisites are present.

        Returns a result dict with keys:
          success      : bool
          steps        : int  (number of env steps issued)
          trajectory   : list of (obs, action_desc, next_obs, succ) tuples
          final_obs    : last wrapped obs
          episode_done : bool
          reason       : str (present on failure, explaining why)
        """
        if not hasattr(env, "step_command"):
            return {"success": False, "steps": 0, "trajectory": [],
                    "final_obs": env.get_obs() if hasattr(env, "get_obs") else {},
                    "episode_done": False,
                    "reason": "env does not expose step_command; "
                              "use MineDojoObsWrapper"}

        obs = env.get_obs() if hasattr(env, "get_obs") else {}

        # ── Check prerequisites ───────────────────────────────────────────────
        raw = env.get_raw_obs() if hasattr(env, "get_raw_obs") else None
        if raw is not None:
            inv_counts = _parse_inv_counts(raw)
        else:
            inv_counts = {}

        prereqs = _CRAFT_PREREQS.get(self.skill_name, [])
        for item_name, min_count in prereqs:
            actual = inv_counts.get(item_name, 0)
            if actual < min_count:
                return {
                    "success":      False,
                    "steps":        0,
                    "trajectory":   [],
                    "final_obs":    obs,
                    "episode_done": False,
                    "reason":       (
                        f"prerequisite not met: need {min_count}x {item_name}, "
                        f"have {actual}"
                    ),
                }

        # ── Execute craft sequence ────────────────────────────────────────────
        sequence = _CRAFT_SEQUENCES.get(self.skill_name, [])
        steps       = 0
        trajectory: list = []
        done        = False

        for action_type, item_name in sequence:
            if steps >= self.cfg.max_steps:
                break
            next_obs, _rew, done, _info = env.step_command(action_type, item_name)
            x_curr = evgs.extract(obs)
            x_next = evgs.extract(next_obs)
            succ_step = EVGS.predicate_holds(x_curr, x_next, self.subgoal)
            trajectory.append((obs, f"{action_type}({item_name})", next_obs, succ_step))
            obs   = next_obs
            steps += 1
            if done:
                break

        # ── Verify success ────────────────────────────────────────────────────
        final_vars = evgs.extract(obs)
        success = bool(final_vars[self.subgoal.var_index] > 0.5)

        result: Dict[str, Any] = {
            "success":      success,
            "steps":        steps,
            "trajectory":   trajectory,
            "final_obs":    obs,
            "episode_done": done,
        }
        if not success:
            result["reason"] = (
                f"craft sequence completed but {self.skill_name} not in inventory"
            )
        return result


# ---------------------------------------------------------------------------
# BCOptionWrapper — BC policy wrapper for gathering skills
# ---------------------------------------------------------------------------

def _load_bc_policy_from_pt(pt_path: str):
    """
    Load a BCPolicy checkpoint from a .pt file.

    The BCPolicy architecture matches pretrain_bc_minerl.py:
      conv(3→32, 8×8, s4) → conv(32→64, 4×4, s2) → conv(64→64, 3×3, s1)
      → flatten → fc(1024→512) → fc(512→n_actions)

    Returns (policy, metadata_dict).
    """
    import torch
    import torch.nn as nn

    class _BCPolicy(nn.Module):
        def __init__(self, n_actions: int = _N_ACTIONS) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            )
            self.n_actions = n_actions

        def forward(self, x):
            return self.fc(self.conv(x))

        def predict(self, obs_hwc: np.ndarray) -> int:
            """obs_hwc: uint8 ndarray (H, W, 3)."""
            self.eval()
            import torch as _torch
            with _torch.no_grad():
                x = (
                    _torch.from_numpy(obs_hwc.astype(np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                return int(self(x).argmax(dim=-1).item())

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    policy = _BCPolicy(n_actions=ckpt.get("n_actions", _N_ACTIONS))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy, ckpt


class BCOptionWrapper(OptionPolicy):
    """
    Thin OptionPolicy wrapper around a BC policy (no SB3 dependency).

    bc_policy must expose .predict(obs_hwc: np.ndarray) -> int
    which takes a (H, W, 3) uint8 image and returns a discrete action index.

    env.step(action_idx) must accept integer indices (MineDojoObsWrapper does).
    """

    def __init__(self, subgoal: Subgoal, cfg: OptionConfig, bc_policy):
        super().__init__(subgoal, cfg)
        self._policy = bc_policy

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        """obs: wrapped dict with 'rgb' key (H, W, 3) uint8."""
        rgb = obs.get("rgb") if isinstance(obs, dict) else None
        if rgb is None or not hasattr(self._policy, "predict"):
            return int(np.random.randint(_N_ACTIONS))
        return int(self._policy.predict(rgb))

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        """Execute BC policy until success, timeout, or env done."""
        obs    = env.get_obs() if hasattr(env, "get_obs") else {}
        steps  = 0
        success = False
        done    = False
        trajectory: list = []

        while steps < self.cfg.max_steps:
            action = self.act(obs)
            next_obs, _rew, done, _info = env.step(action)

            x_curr = evgs.extract(obs)
            x_next = evgs.extract(next_obs)
            succ   = EVGS.predicate_holds(x_curr, x_next, self.subgoal)
            trajectory.append((obs, action, next_obs, succ))
            obs   = next_obs
            steps += 1

            if succ:
                success = True
                if self.cfg.terminate_on_success:
                    break
            if done:
                break

        return {
            "success":      success,
            "steps":        steps,
            "trajectory":   trajectory,
            "final_obs":    obs,
            "episode_done": done,
        }


# ---------------------------------------------------------------------------
# ScriptedGatherOption — rotate-and-attack to collect a target item
# ---------------------------------------------------------------------------

# MineDojo NNActionSpaceWrapper 8-dim action indices (same cam_interval=15)
# [fb, lr, jss, pitch_bin, yaw_bin, fn, craft_arg, slot]
# noop bins: pitch=12, yaw=12   (bin*15 - 180 = 0 degrees)
# fn: 0=noop, 1=use, 2=drop, 3=attack, 4=craft, 5=equip, 6=place, 7=destroy
_ACT_NOOP        = np.array([0, 0, 0, 12, 12, 0, 0, 0], dtype=np.int64)
_ACT_ATTACK      = np.array([0, 0, 0, 12, 12, 3, 0, 0], dtype=np.int64)
_ACT_FWD_ATTACK  = np.array([1, 0, 0, 12, 12, 3, 0, 0], dtype=np.int64)
_ACT_FWD         = np.array([1, 0, 0, 12, 12, 0, 0, 0], dtype=np.int64)
_ACT_LOOK_DOWN   = np.array([0, 0, 0, 13, 12, 0, 0, 0], dtype=np.int64)  # pitch +15°
_ACT_LOOK_UP     = np.array([0, 0, 0, 11, 12, 0, 0, 0], dtype=np.int64)  # pitch -15°

def _yaw_bin(delta_deg: int) -> np.ndarray:
    """Return 8-dim action that rotates yaw by delta_deg (multiple of 15)."""
    yaw_bin = 12 + int(delta_deg // 15)          # 12 = noop (0°), +1 = +15°, -1 = -15°
    yaw_bin = max(0, min(24, yaw_bin))
    return np.array([0, 0, 0, 12, yaw_bin, 0, 0, 0], dtype=np.int64)


class ScriptedGatherOption(OptionPolicy):
    """
    Scripted rotate-and-attack option for gathering resources in MineDojo.

    Strategy:
      Phase 1 — Scan:  rotate yaw 360° in 15° steps while attacking each position
                        for `attack_per_angle` steps.  Stop early if item acquired.
      Phase 2 — Forward push:  move+attack for `fwd_steps` to close distance to block.
      Repeat phases until success, timeout, or env done.

    Works reliably when the target resource is within 4 blocks of spawn because
    the 360° scan will face each direction and break any reachable block.

    Uses the MinedojoObsWrapper integer action index interface (DIA 25-action space)
    or, if `env._DIA_TO_MD` is absent, falls back to raw 8-dim numpy arrays.
    """

    # Target item names in MineDojo inventory for each DIA skill
    _SKILL_ITEMS: Dict[str, list] = {
        "wood":    ["oak_log", "log", "birch_log", "spruce_log", "jungle_log", "acacia_log"],
        "stone":   ["cobblestone", "stone"],
        "coal":    ["coal"],
        "ironore": ["iron_ore"],
        "diamond": ["diamond"],
    }

    def __init__(self, subgoal: Subgoal, cfg: OptionConfig, skill_name: str,
                 attack_per_angle: int = 8, fwd_steps: int = 12,
                 n_scan_rounds: int = 3):
        super().__init__(subgoal, cfg)
        self.skill_name = skill_name
        self.target_items = self._SKILL_ITEMS.get(skill_name, [skill_name])
        self.attack_per_angle = attack_per_angle
        self.fwd_steps = fwd_steps
        self.n_scan_rounds = n_scan_rounds

    def act(self, obs) -> int:
        return 6  # attack (DIA action 6)

    def _get_target_count(self, env) -> int:
        """Return total count of target items in current inventory."""
        raw = env._last_obs if hasattr(env, "_last_obs") else None
        if raw is None and hasattr(env, "get_raw_obs"):
            raw = env.get_raw_obs()
        if raw is None:
            return 0
        counts = _parse_inv_counts(raw)
        return sum(counts.get(item, 0) for item in self.target_items)

    def _step(self, env, action: np.ndarray):
        """Step env with raw 8-dim action, bypassing DIA integer conversion."""
        # Directly call the wrapped env's step to avoid the _DIA_TO_MD conversion
        result = env.env.step(action)
        if len(result) == 5:
            raw, rew, term, trunc, info = result
            done = term or trunc
        else:
            raw, rew, done, info = result
        # Update the wrapper's cached obs
        env._last_obs = raw
        wrapped_obs = env._convert(raw) if hasattr(env, "_convert") else {}
        return wrapped_obs, float(rew), bool(done), info

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        obs    = env.get_obs() if hasattr(env, "get_obs") else {}
        steps  = 0
        success = False
        done    = False
        trajectory: list = []
        initial_count = self._get_target_count(env)

        for _round in range(self.n_scan_rounds):
            # ── Scan phase: rotate 360° attacking at each angle ─────────────
            for _angle_step in range(24):          # 24 × 15° = 360°
                yaw_act = _yaw_bin(15)             # rotate right 15°
                obs, _, done, _ = self._step(env, yaw_act)
                steps += 1
                if steps >= self.cfg.max_steps or done:
                    break

                # Attack at this angle for attack_per_angle steps
                for _ in range(self.attack_per_angle):
                    if steps >= self.cfg.max_steps or done:
                        break
                    obs, _rew, done, _ = self._step(env, _ACT_FWD_ATTACK)
                    x_curr = evgs.extract(obs)
                    steps += 1

                    current = self._get_target_count(env)
                    if current > initial_count:
                        succ_step = True
                        trajectory.append((obs, "fwd_attack", obs, succ_step))
                        success = True
                        if self.cfg.terminate_on_success:
                            return {
                                "success":      True,
                                "steps":        steps,
                                "trajectory":   trajectory,
                                "final_obs":    obs,
                                "episode_done": done,
                            }

                if done:
                    break

            if done or (success and self.cfg.terminate_on_success):
                break

            # ── Forward push phase ──────────────────────────────────────────
            for _ in range(self.fwd_steps):
                if steps >= self.cfg.max_steps or done:
                    break
                obs, _rew, done, _ = self._step(env, _ACT_FWD_ATTACK)
                steps += 1
                current = self._get_target_count(env)
                if current > initial_count:
                    success = True
                    if self.cfg.terminate_on_success:
                        break

        return {
            "success":      success,
            "steps":        steps,
            "trajectory":   trajectory,
            "final_obs":    obs,
            "episode_done": done,
        }


# ---------------------------------------------------------------------------
# load_bc_option — convenience loader for BC-warm-started options
# ---------------------------------------------------------------------------

def load_bc_option(skill: str, pt_path: str,
                   cfg: OptionConfig) -> Optional["BCOptionWrapper"]:
    """
    Load a BC policy checkpoint and wrap it as a BCOptionWrapper.

    Parameters
    ----------
    skill   : DIA variable name ('wood', 'stone', etc.)
    pt_path : path to .pt file saved by pretrain_bc_minerl.py (or equivalent)
    cfg     : OptionConfig

    Returns None if the file is missing or loading fails.
    """
    from .evgs_minedojo import VAR_NAMES as _VAR_NAMES
    from .types import Predicate as _Predicate

    if not __import__("os").path.exists(pt_path):
        return None
    try:
        var_names = list(_VAR_NAMES)
        if skill not in var_names:
            raise ValueError(f"Unknown skill: {skill!r}; expected one of {var_names}")
        var_idx = var_names.index(skill)
        sg      = Subgoal(var_index=var_idx, predicate=_Predicate.UP)
        policy, meta = _load_bc_policy_from_pt(pt_path)
        acc_str = (f"{meta.get('train_acc', 0.0):.3f}"
                   if isinstance(meta.get("train_acc"), float) else "?")
        print(f"  [{skill}] BC policy loaded  "
              f"(train_acc={acc_str}  epochs={meta.get('epochs', '?')})")
        return BCOptionWrapper(sg, cfg, policy)
    except Exception as exc:
        print(f"  [{skill}] WARNING: could not load BC policy from {pt_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Gather skill action set (10-action subset for voxel-conditioned PPO)
# ---------------------------------------------------------------------------

# Maps discrete index 0-9 to MineDojo 8-dim MultiDiscrete array.
# [fb, lr, jss, pitch_bin, yaw_bin, fn, craft_arg, slot]
# Noop bins: pitch=12, yaw=12 (bin*15-180=0 degrees)
# fn: 0=noop, 1=use, 2=drop, 3=attack, 4=craft, 5=equip, 6=place, 7=destroy
GATHER_ACTIONS = np.array([
    [0, 0, 0, 12, 12, 0, 0, 0],  # 0: noop
    [1, 0, 0, 12, 12, 0, 0, 0],  # 1: forward
    [0, 0, 0, 12, 12, 3, 0, 0],  # 2: attack
    [1, 0, 0, 12, 12, 3, 0, 0],  # 3: forward + attack
    [0, 0, 0, 12, 13, 0, 0, 0],  # 4: yaw right +15 degrees
    [0, 0, 0, 12, 11, 0, 0, 0],  # 5: yaw left  -15 degrees
    [0, 0, 0, 13, 12, 0, 0, 0],  # 6: look down +15 degrees pitch
    [0, 0, 0, 11, 12, 0, 0, 0],  # 7: look up   -15 degrees pitch
    [1, 0, 1, 12, 12, 0, 0, 0],  # 8: forward + jump
    [1, 0, 3, 12, 12, 3, 0, 0],  # 9: sprint + forward + attack
], dtype=np.int64)

N_GATHER_ACTIONS = len(GATHER_ACTIONS)


# ---------------------------------------------------------------------------
# VoxelGatherWrapper: adds target_voxel + agent_state to observation
# ---------------------------------------------------------------------------

class VoxelGatherWrapper(MineDojoObsWrapper):
    """
    Extends MineDojoObsWrapper with voxel-based target block observations
    for training gather skill policies with PPO.

    Observation dict keys:
      "target_voxel"  (N,) float32 binary: 1 where voxel contains a target block
      "agent_state"   (6,) float32 normalized: [pitch, yaw, health, food, on_ground, has_tool]

    Action space: Discrete(10) mapped through GATHER_ACTIONS to 8-dim MineDojo arrays.

    Parameters
    ----------
    env         : raw MineDojo environment (with use_voxel=True)
    evgs        : EVGS instance from make_minedojo_evgs()
    skill_name  : one of SKILL_TARGETS keys ('wood', 'stone', 'coal', 'ironore', 'diamond')
    tool_var_idx: index in the EVGS inventory vector for the required tool (-1 if hand only)
    voxel_size  : voxel grid dimensions passed to minedojo.make(); default (7, 3, 7)
    """

    def __init__(self, env, evgs: "EVGS", skill_name: str,
                 tool_var_idx: int = -1,
                 voxel_size: tuple = (7, 3, 7)):
        super().__init__(env, evgs)
        self.skill_name    = skill_name
        self.tool_var_idx  = tool_var_idx
        self._voxel_size   = voxel_size
        self._voxel_dim    = int(voxel_size[0]) * int(voxel_size[1]) * int(voxel_size[2])

        # Import lazily to avoid circular imports at module load time
        from .evgs_minedojo import voxel_extract_target as _vet, get_agent_state as _gas
        self._voxel_extract_target = _vet
        self._get_agent_state      = _gas

        # Override gym spaces for SB3
        try:
            import gym
            from gym import spaces as _spaces
            self.observation_space = _spaces.Dict({
                "target_voxel": _spaces.Box(0.0, 1.0, (self._voxel_dim,), dtype=np.float32),
                "agent_state":  _spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
            })
            self.action_space = _spaces.Discrete(N_GATHER_ACTIONS)
        except Exception:
            pass

    # -- Override step to translate Discrete(10) to 8-dim array --

    def step(self, action_idx: int):
        """Accept Discrete(10) index, translate to 8-dim, step raw env."""
        idx = int(action_idx) % N_GATHER_ACTIONS
        act8 = GATHER_ACTIONS[idx]
        result = self.env.step(act8)
        if len(result) == 5:
            raw, rew, term, trunc, info = result
            done = term or trunc
        else:
            raw, rew, done, info = result
        self._last_raw = raw
        return self._convert(raw), float(rew), bool(done), info

    # -- Override _convert to inject voxel + agent_state --

    def _convert(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        target_voxel = self._voxel_extract_target(raw_obs, self.skill_name)
        # Pad or truncate to expected size in case voxel shape differs
        if target_voxel.shape[0] != self._voxel_dim:
            padded = np.zeros(self._voxel_dim, dtype=np.float32)
            n = min(target_voxel.shape[0], self._voxel_dim)
            padded[:n] = target_voxel[:n]
            target_voxel = padded

        agent_state = self._get_agent_state(raw_obs).copy()
        # Fill has_tool slot (index 5) from inventory vector
        if self.tool_var_idx >= 0:
            inv = self.evgs.extract(raw_obs)
            agent_state[5] = float(inv[self.tool_var_idx] > 0.5)

        return {
            "target_voxel": target_voxel,
            "agent_state":  agent_state,
        }

    def get_obs(self) -> Dict[str, np.ndarray]:
        if self._last_raw is not None:
            return self._convert(self._last_raw)
        return {
            "target_voxel": np.zeros(self._voxel_dim, dtype=np.float32),
            "agent_state":  np.zeros(6, dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# VoxelRewardWrapper: proximity + acquisition reward shaping
# ---------------------------------------------------------------------------

class VoxelRewardWrapper:
    """
    Reward shaping wrapper for gather skill training.

    Wraps a VoxelGatherWrapper and adds three reward signals:

    1. Proximity delta: +scale * (prev_dist - curr_dist) per step when target
       block is visible in the voxel grid.  Encourages navigation toward target.
    2. Acquisition: +1.0 when target item count in inventory increases.
    3. Time penalty: -0.005 per step (encourages efficiency).

    Distance is the L1 distance (in voxel grid coordinates) to the nearest
    target-block cell. When no target is visible the proximity signal is 0.

    Parameters
    ----------
    env             : VoxelGatherWrapper instance
    skill_name      : gather skill name (matches SKILL_TARGETS keys)
    evgs            : EVGS instance
    proximity_scale : reward scaling for proximity delta (default 0.1)
    acq_reward      : reward for acquiring one unit of target item (default 1.0)
    time_penalty    : per-step penalty (default -0.005)
    """

    def __init__(self, env: VoxelGatherWrapper, skill_name: str, evgs: "EVGS",
                 proximity_scale: float = 0.1,
                 acq_reward: float = 1.0,
                 time_penalty: float = -0.005):
        self.env             = env
        self.skill_name      = skill_name
        self.evgs            = evgs
        self.proximity_scale = proximity_scale
        self.acq_reward      = acq_reward
        self.time_penalty    = time_penalty

        self._prev_dist      = self._max_dist()
        self._prev_count     = 0

        # Forward gym spaces
        self.observation_space = env.observation_space
        self.action_space      = env.action_space

        # Target item names (from ScriptedGatherOption._SKILL_ITEMS)
        self._target_items = ScriptedGatherOption._SKILL_ITEMS.get(skill_name, [skill_name])

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = obs[0] if isinstance(obs, tuple) else obs
        self._prev_dist  = self._dist_from_obs(obs)
        self._prev_count = self._inv_count()
        return obs

    def step(self, action):
        obs, ext_rew, done, info = self.env.step(action)

        curr_dist  = self._dist_from_obs(obs)
        curr_count = self._inv_count()

        shaped = self.time_penalty

        # Proximity signal (only when target visible)
        if self._prev_dist < self._max_dist() or curr_dist < self._max_dist():
            delta = self._prev_dist - curr_dist
            shaped += float(np.clip(self.proximity_scale * delta, -0.2, 0.5))

        # Acquisition reward
        if curr_count > self._prev_count:
            shaped += self.acq_reward * (curr_count - self._prev_count)

        self._prev_dist  = curr_dist
        self._prev_count = curr_count

        return obs, float(ext_rew + shaped), bool(done), info

    # Passthrough
    def get_obs(self):
        return self.env.get_obs()

    def get_raw_obs(self):
        return self.env.get_raw_obs()

    def step_command(self, action_type, item_name):
        return self.env.step_command(action_type, item_name)

    # -- Helpers --

    def _max_dist(self) -> float:
        """Max possible L1 distance in voxel grid (used as sentinel for no target)."""
        vs = self.env._voxel_size
        return float(vs[0] + vs[1] + vs[2])

    def _dist_from_obs(self, obs: Dict[str, np.ndarray]) -> float:
        """L1 distance to nearest target cell in flattened target_voxel."""
        tv = obs.get("target_voxel", None)
        if tv is None or tv.sum() == 0:
            return self._max_dist()
        vs = self.env._voxel_size
        # Reshape to 3D (x, y, z) and find nearest 1-cell
        try:
            grid = tv.reshape(vs)
            coords = np.argwhere(grid > 0.5)
            if len(coords) == 0:
                return self._max_dist()
            # Agent is at center of grid
            center = np.array([vs[0] // 2, vs[1] // 2, vs[2] // 2])
            dists  = np.abs(coords - center).sum(axis=1)
            return float(dists.min())
        except Exception:
            return self._max_dist()

    def _inv_count(self) -> int:
        raw = self.env.get_raw_obs()
        if raw is None:
            return 0
        counts = _parse_inv_counts(raw)
        return sum(counts.get(item, 0) for item in self._target_items)


# ---------------------------------------------------------------------------
# VoxelGatherOption: SB3 PPO policy option for gather skills
# ---------------------------------------------------------------------------

class VoxelGatherOption(OptionPolicy):
    """
    Executes a pre-trained SB3 PPO gather policy that operates on voxel
    observations (target_voxel + agent_state) to navigate toward and mine
    a target block.

    Usage
    -----
    env must be a VoxelGatherWrapper (provides target_voxel + agent_state obs,
    Discrete(10) action space).

    model_path : path to a .zip file saved by scripts/train_gather_skill.py
                 (SB3 PPO with MultiInputPolicy).

    Parameters
    ----------
    subgoal    : Subgoal(var_index, predicate=UP) for the target EVGS variable
    cfg        : OptionConfig (max_steps, terminate_on_success)
    model_path : path to SB3 PPO .zip, or a pre-loaded SB3 model instance
    skill_name : gather skill name for logging
    """

    def __init__(self, subgoal: "Subgoal", cfg: "OptionConfig",
                 model_path, skill_name: str = ""):
        super().__init__(subgoal, cfg)
        self.skill_name = skill_name
        self._model     = None

        if SB3_OK and model_path is not None:
            if isinstance(model_path, str):
                self._model = SB3PPO.load(model_path)
            else:
                self._model = model_path   # pre-loaded

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        """Predict action using PPO policy; fallback to attack (idx 2) if no model."""
        if self._model is None:
            return 2  # attack
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)

    def run(self, env, evgs: "EVGS") -> Dict[str, Any]:
        """
        Execute PPO gather policy until subgoal achieved or max_steps exceeded.

        env must be a VoxelGatherWrapper or compatible (exposes get_obs() returning
        {'target_voxel': ..., 'agent_state': ...}).
        """
        if not hasattr(env, "get_obs"):
            raise TypeError("VoxelGatherOption.run requires env with get_obs()")

        obs     = env.get_obs()
        steps   = 0
        success = False
        done    = False
        trajectory: list = []

        while steps < self.cfg.max_steps:
            action = self.act(obs)

            result = env.step(action)
            if len(result) == 5:
                next_obs, _rew, term, trunc, _info = result
                done = term or trunc
            else:
                next_obs, _rew, done, _info = result

            x_curr = evgs.extract(obs)
            x_next = evgs.extract(next_obs)
            succ   = EVGS.predicate_holds(x_curr, x_next, self.subgoal)

            trajectory.append((obs, action, next_obs, succ))
            obs   = next_obs
            steps += 1

            if succ:
                success = True
                if self.cfg.terminate_on_success:
                    break
            if done:
                break

        return {
            "success":      success,
            "steps":        steps,
            "trajectory":   trajectory,
            "final_obs":    obs,
            "episode_done": done,
        }

