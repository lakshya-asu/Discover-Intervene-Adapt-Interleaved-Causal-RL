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
# camera: [pitch_delta, yaw_delta] in degrees (positive pitch = look down)
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
    [],                                                          # 0:  noop
    [("forward", 1)],                                            # 1:  forward
    [("back", 1)],                                               # 2:  back
    [("left", 1)],                                               # 3:  strafe left
    [("right", 1)],                                              # 4:  strafe right
    [("jump", 1)],                                               # 5:  jump
    [("attack", 1)],                                             # 6:  attack (mine/chop)
    [("use", 1)],                                                # 7:  use (place/craft/open)
    [("forward", 1), ("attack", 1)],                             # 8:  forward + attack
    [("forward", 1), ("jump", 1)],                               # 9:  forward + jump
    [("forward", 1), ("sprint", 1)],                             # 10: sprint forward
    [("camera", np.array([-15.0, 0.0]))],                        # 11: camera up  (coarse)
    [("camera", np.array([15.0, 0.0]))],                         # 12: camera down (coarse)
    [("camera", np.array([0.0, -15.0]))],                        # 13: camera left (coarse)
    [("camera", np.array([0.0, 15.0]))],                         # 14: camera right (coarse)
    # ── additional actions for effective 3D skill execution ──────────────────
    [("forward", 1), ("sprint", 1), ("attack", 1)],              # 15: sprint + attack (aggressive)
    [("camera", np.array([-5.0, 0.0]))],                         # 16: camera up   (fine)
    [("camera", np.array([5.0, 0.0]))],                          # 17: camera down  (fine)
    [("camera", np.array([0.0, -5.0]))],                         # 18: camera left  (fine)
    [("camera", np.array([0.0, 5.0]))],                          # 19: camera right (fine)
    [("hotbar.1", 1)],                                           # 20: equip slot 1
    [("hotbar.2", 1)],                                           # 21: equip slot 2
    [("forward", 1), ("use", 1)],                                # 22: forward + use (walk + interact)
    [("jump", 1), ("attack", 1)],                                # 23: jump + attack
    [("forward", 1), ("jump", 1), ("attack", 1)],                # 24: forward + jump + attack
]
N_ACTIONS = len(_ACTION_DEFS)

# ── Per-skill preferred action indices for biased PPO exploration ─────────────
# Actions most likely to produce reward signal for each skill.
# Used by BiasedDiscrete to make rollout collection more sample-efficient.
SKILL_PREFERRED_ACTIONS: Dict[str, List[int]] = {
    # Gathering skills: physical actions dominate
    "wood":        [6, 8, 11, 15, 24, 9, 14, 13, 16, 17],  # attack, fwd+attack, look up, sprint+atk
    "stone":       [6, 12, 17, 8, 1, 11, 16, 14, 13],       # attack, look down, fwd+attack
    "coal":        [6, 8, 11, 14, 13, 9, 15, 16, 17, 1],    # attack + exploration
    "ironore":     [6, 8, 12, 17, 14, 13, 9, 15, 24, 1],    # attack + look down + exploration
    "diamond":     [6, 8, 12, 17, 14, 13, 9, 15, 24, 1],    # same as ironore (mine deep)
    # Crafting/smelting skills: interaction actions dominate
    "furnace":     [7, 6, 12, 17, 11, 16, 1, 22, 20, 21],   # use, look down, interact
    "stonepickaxe":[7, 6, 12, 17, 11, 16, 22, 20, 21, 1],   # same as furnace
    "iron":        [7, 12, 17, 11, 16, 6, 22, 1, 20, 21],   # use (smelt), look down
    "ironpickaxe": [7, 6, 12, 17, 11, 16, 22, 20, 21, 1],   # same as furnace
}


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
# SkillScriptedOption — task-specific macro behaviors (much better than random)
# ---------------------------------------------------------------------------

class SkillScriptedOption(OptionPolicy):
    """
    Scripted option that cycles a task-specific action macro.

    Each skill has a handcrafted macro sequence designed around the physical
    actions needed to acquire that resource in 3D Minecraft:
      - Gathering skills (wood/stone/coal/ironore/diamond): navigation + attack
      - Crafting/smelting skills: interaction attempts via use/inventory

    The macro is cycled indefinitely (looped) until success or max_steps.
    A small random perturbation is added to camera at each step to avoid
    getting stuck looking at the same spot.
    """

    # Action macro sequences per skill (indices into _ACTION_DEFS)
    # Legend (key actions): 6=attack, 8=fwd+attack, 11=cam_up, 12=cam_down,
    #   13=cam_left, 14=cam_right, 15=sprint+attack, 24=fwd+jump+attack,
    #   7=use, 16=cam_up_fine, 17=cam_down_fine, 9=fwd+jump
    _MACROS: Dict[str, List[int]] = {
        # WOOD: approach tree trunk, look up to hit trunk, chop continuously.
        # Trees are 2-4 blocks tall; look slightly up, sprint+attack while walking.
        "wood": [
            11, 11,                    # look up (tree trunk is above ground level)
            8, 8, 8, 8, 8,             # forward + attack (walk into tree)
            6, 6, 6, 6, 6, 6,          # attack in place (break trunk block)
            15,                        # sprint forward (close gap)
            8, 8, 8, 8,                # forward + attack
            9,                         # jump forward (obstacle)
            11,                        # look up a bit more (top of trunk)
            6, 6, 6, 6, 6,             # attack
            14, 14, 14,                # turn right (find another tree)
            8, 8, 8,                   # approach
            13, 13, 13, 13,            # turn left (zig-zag search)
            8, 8, 8, 8, 8,             # forward + attack
            24,                        # fwd + jump + attack
        ],
        # STONE: mine cobblestone. Stone is at ground level / in walls.
        # Look down to aim at ground stone, attack repeatedly.
        "stone": [
            12, 12,                    # look down (aim at ground block)
            6, 6, 6, 6, 6, 6, 6, 6,   # attack x8 (mine cobblestone)
            11,                        # look up a bit
            1,                         # step forward (new block)
            12,                        # look down
            6, 6, 6, 6, 6, 6,          # attack x6
            11, 11,                    # look forward
            14, 14,                    # turn right (try cliff face)
            12, 12,                    # look down
            6, 6, 6, 6, 6,             # attack
            11, 11,                    # look forward
            8, 8,                      # move forward
        ],
        # COAL: find coal ore in stone walls / cave entrances.
        # Walk forward attacking rock faces; turn to search.
        "coal": [
            11,                        # look up slightly (coal often in walls at eye height)
            8, 8, 8, 8, 8,             # forward + attack
            6, 6, 6, 6,                # attack wall
            14, 14,                    # turn right
            8, 8, 8,                   # forward + attack
            13, 13, 13, 13,            # turn left
            8, 8, 8, 8,                # forward + attack
            6, 6, 6, 6,                # attack
            9,                         # jump + forward
            14, 14, 14,                # turn right (search)
            6, 6, 6, 6, 6,             # attack
        ],
        # IRONORE: iron ore is underground (y=15-63). Same search pattern.
        "ironore": [
            8, 8, 8, 8, 8,             # forward + attack (cave)
            12, 12,                    # look down slightly (deeper ore)
            6, 6, 6, 6, 6, 6,          # attack
            11, 11,                    # look forward
            14, 14,                    # turn right
            8, 8, 8,                   # forward + attack
            13, 13, 13,                # turn left
            6, 6, 6, 6,                # attack
            9,                         # jump
            15,                        # sprint + attack
            12,                        # look down
            6, 6, 6, 6, 6,             # attack
            11,                        # look up
        ],
        # FURNACE: craft from 8 cobblestone at crafting table.
        # Can't script GUI crafting — try use + inventory interactions.
        "furnace": [
            7, 7, 7,                   # use (right-click: open crafting table/interact)
            12,                        # look down
            7, 7,                      # use
            11,                        # look up
            6, 6,                      # attack (break any block in the way)
            14,                        # turn right
            7, 7, 7,                   # use
            1,                         # forward
            7, 7,                      # use
            12,                        # look down
            7, 7,                      # use
            11,                        # look up
            8, 8,                      # forward + attack
        ],
        # STONEPICKAXE: craft from sticks + cobblestone.
        "stonepickaxe": [
            7, 7, 7, 7,                # use repeatedly (crafting attempt)
            12,                        # look down
            7, 7,                      # use
            11,                        # look up
            7, 7,                      # use
            14, 14,                    # turn
            7, 7, 7,                   # use
            22,                        # forward + use (walk to crafting table)
            7, 7,                      # use
        ],
        # IRON: smelt iron ore in furnace. Need use near furnace.
        "iron": [
            7, 7, 7, 7, 7,             # use (interact with furnace)
            12,                        # look down
            7, 7, 7,                   # use
            11,                        # look up
            7, 7,                      # use
            6,                         # attack
            22,                        # forward + use
            7, 7, 7,                   # use
        ],
        # IRONPICKAXE: craft from iron ingots + sticks.
        "ironpickaxe": [
            7, 7, 7, 7,                # use
            12,                        # look down
            7, 7,                      # use
            11,                        # look up
            7, 7, 7,                   # use
            14, 14,                    # turn
            7, 7, 7,                   # use
            22,                        # forward + use
            7, 7,                      # use
        ],
        # DIAMOND: mine diamond ore with iron pickaxe (y=5-12).
        # Same as ironore: walk + attack walls at lower levels.
        "diamond": [
            12, 12,                    # look down (go deeper)
            6, 6, 6, 6, 6, 6, 6,       # attack (dig down)
            11, 11,                    # look forward
            8, 8, 8, 8,                # forward + attack
            12,                        # look down
            6, 6, 6, 6, 6, 6,          # attack (mine wall)
            11,                        # look up
            14, 14,                    # turn right
            8, 8, 8,                   # forward + attack
            13, 13,                    # turn left
            6, 6, 6, 6, 6,             # attack
            9,                         # jump
        ],
    }
    _DEFAULT_MACRO = [8, 8, 8, 8, 6, 6, 6, 6, 14, 14, 6, 6, 13, 13, 9]

    def __init__(self, subgoal: Subgoal, cfg: OptionConfig, var_name: str,
                 noise_prob: float = 0.1):
        """
        Args:
            var_name:   DIA variable name ('wood', 'stone', etc.)
            noise_prob: probability of replacing a macro step with a random action
                        (adds exploration so the agent doesn't get stuck in loops)
        """
        super().__init__(subgoal, cfg)
        self.var_name   = var_name
        self.noise_prob = noise_prob
        self._macro     = self._MACROS.get(var_name, self._DEFAULT_MACRO)
        self._step      = 0

    def act(self, obs) -> int:
        if self.noise_prob > 0 and np.random.random() < self.noise_prob:
            # Small exploration: random camera tweak or movement
            return int(np.random.choice([0, 1, 5, 13, 14, 16, 17, 18, 19]))
        idx = self._macro[self._step % len(self._macro)]
        self._step += 1
        return int(idx)

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        obs     = env.get_obs() if hasattr(env, "get_obs") else {}
        self._step = 0
        steps   = 0
        success = False
        trajectory: list = []

        while steps < self.cfg.max_steps:
            action                    = self.act(obs)
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
# BiasedDiscrete — Discrete space that biases sample() toward useful actions.
# Used during PPO training so rollout collection hits reward more often.
# ---------------------------------------------------------------------------

if GYM_OK and spaces is not None:
    class BiasedDiscrete(spaces.Discrete):
        """
        Discrete(N) action space whose sample() returns preferred actions
        with probability `bias_prob`, falling back to uniform otherwise.

        This is used in train_minerl_skill.py: when SB3 PPO calls
        env.action_space.sample() during rollout collection (exploration),
        it preferentially samples skill-appropriate actions, making early
        reward signals much more likely.
        """
        def __init__(self, n: int, preferred: List[int], bias_prob: float = 0.7):
            super().__init__(n)
            self.preferred  = [int(a) for a in preferred if 0 <= int(a) < n]
            self.bias_prob  = float(bias_prob)

        def sample(self, mask=None) -> int:  # type: ignore[override]
            if self.preferred and np.random.random() < self.bias_prob:
                return int(np.random.choice(self.preferred))
            return int(super().sample(mask=mask) if mask is not None
                       else super().sample())
else:
    BiasedDiscrete = None  # type: ignore


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
