# src/dia/primitives_minerl.py
"""
Hierarchical primitive options for MineRL 3D Minecraft.

Each high-level skill (e.g. "acquire wood") is decomposed into a chain of
primitive PPO options.  The chain is skill-specific: not all primitives are
valid or useful for every item.

Primitive vocabulary
--------------------
  approach         — navigate to within interaction range of the target
  aim              — place the crosshair on the target block face
  attack           — BREAK the target block (wood, stone, ores, diamond)
  interact         — RIGHT-CLICK on the target (furnace, crafting table)
  navigate_deep    — descend underground until ore depth is reached (diamond)

Skill → primitive chain mapping (SKILL_PRIMITIVE_CHAIN)
-------------------------------------------------------
  wood        : approach → aim → attack
  stone       : approach → aim → attack
  coal        : approach → aim → attack
  ironore     : approach → aim → attack
  diamond     : approach → navigate_deep → aim → attack
  furnace     : approach → aim → interact       (right-click to open GUI)
  stonepickaxe: approach → aim → interact       (open crafting table)
  iron        : approach → aim → interact       (open furnace, place items)
  ironpickaxe : approach → aim → interact       (open crafting table)

Validity rule
-------------
Only the primitives listed in SKILL_PRIMITIVE_CHAIN[var_name] are trained
and executed for that skill.  CompositeSkillOption reads this map to know
which stage models to load and in what order.

CLIP texts per stage
--------------------
PRIMITIVE_TEXTS[stage][var_name] — visually grounded description of what
the agent *sees* while successfully executing that stage.  Used for:
  * CLIP dense reward during PPO training (text embedding → cosine similarity)
  * VLM override at inference (scene-specific instruction replaces default text)

Training
--------
  conda run -n dia-minecraft python scripts/train_primitive.py \\
      --var wood --stage approach --steps 200000
  conda run -n dia-minecraft python scripts/train_primitive.py \\
      --var wood --stage aim --steps 100000
  conda run -n dia-minecraft python scripts/train_primitive.py \\
      --var wood --stage attack --steps 300000

  # For diamond (4 stages):
  for stage in approach navigate_deep aim attack; do
    conda run -n dia-minecraft python scripts/train_primitive.py \\
        --var diamond --stage $stage
  done

Output directory layout (models/minerl/):
  approach_{var}.zip
  aim_{var}.zip
  attack_{var}.zip
  interact_{var}.zip          (crafting/smelting skills)
  navigate_deep_{var}.zip     (diamond only)
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

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

from .options_minerl import (
    MineRLObsWrapper, N_ACTIONS, GOAL_DIM, INV_DIM, OBS_H, OBS_W,
    BiasedDiscrete, idx_to_minerl_action,
)


# ── All primitive stage names ──────────────────────────────────────────────────
ALL_STAGES = ("approach", "aim", "attack", "interact", "navigate_deep")

# ── Per-skill valid primitive chain ───────────────────────────────────────────
# Defines which stages are executed (in order) for each DIA variable.
# Only stages listed here are trained and loaded for that skill.
SKILL_PRIMITIVE_CHAIN: Dict[str, List[str]] = {
    "wood":         ["approach", "aim", "attack"],
    "stone":        ["approach", "aim", "attack"],
    "coal":         ["approach", "aim", "attack"],
    "ironore":      ["approach", "aim", "attack"],
    "diamond":      ["approach", "navigate_deep", "aim", "attack"],
    "furnace":      ["approach", "aim", "interact"],
    "stonepickaxe": ["approach", "aim", "interact"],
    "iron":         ["approach", "aim", "interact"],
    "ironpickaxe":  ["approach", "aim", "interact"],
}


# ── CLIP texts per stage per target ───────────────────────────────────────────
PRIMITIVE_TEXTS: Dict[str, Dict[str, str]] = {

    "approach": {
        "wood":         "walking through a Minecraft forest toward a nearby oak tree trunk",
        "stone":        "walking toward a grey stone cliff face or cave entrance in Minecraft",
        "coal":         "walking through a Minecraft cave toward dark black coal ore in stone walls",
        "ironore":      "walking through a cave toward reddish-brown iron ore embedded in stone",
        "furnace":      "walking toward a stone furnace block placed on the ground in Minecraft",
        "stonepickaxe": "walking toward a wooden crafting table in a Minecraft house",
        "iron":         "walking toward a lit furnace with glowing orange fire inside",
        "ironpickaxe":  "walking toward a wooden crafting table to craft a pickaxe",
        "diamond":      "walking through a Minecraft cave toward the deeper underground tunnels",
    },

    "navigate_deep": {
        "diamond": (
            "digging downward through stone in a Minecraft cave to reach deep underground "
            "diamond level, dark stone walls on all sides, no sky visible"
        ),
    },

    "aim": {
        "wood":         "Minecraft first-person view with crosshair aimed at tree trunk wood bark",
        "stone":        "crosshair aimed at a grey stone or cobblestone block face in Minecraft",
        "coal":         "crosshair aimed at a dark coal ore block face embedded in stone",
        "ironore":      "crosshair aimed at reddish-brown iron ore block face in a cave wall",
        "furnace":      "crosshair aimed at a placed stone furnace block face in Minecraft",
        "stonepickaxe": "crosshair aimed at a crafting table block face ready to open menu",
        "iron":         "crosshair aimed at a furnace block face with fire visible inside",
        "ironpickaxe":  "crosshair aimed at a wooden crafting table block face",
        "diamond":      "crosshair aimed at blue diamond ore block face deep underground",
    },

    "attack": {
        "wood":    "chopping a Minecraft tree trunk with bare hands, wood cracking and particles",
        "stone":   "mining grey stone blocks with a pickaxe, rock dust particles flying",
        "coal":    "mining black coal ore with a pickaxe, dark dust particles",
        "ironore": "mining reddish-brown iron ore with a stone pickaxe, sparks flying",
        "diamond": "mining blue diamond ore with an iron pickaxe deep underground, blue sparkles",
    },

    "interact": {
        "furnace":      "opening a stone furnace GUI interface with smelting slots visible",
        "stonepickaxe": "Minecraft crafting table recipe grid showing stone pickaxe pattern",
        "iron":         "placing iron ore and coal into a furnace smelting interface",
        "ironpickaxe":  "Minecraft crafting table recipe grid showing iron pickaxe pattern",
    },
}

# ── Preferred actions per stage (for BiasedDiscrete exploration) ──────────────
# Action index legend (options_minerl.py _ACTION_DEFS):
#   0=noop 1=forward 2=back 3=left 4=right 5=jump 6=attack 7=use
#   8=fwd+attack 9=fwd+jump 10=sprint 11=cam_up(15) 12=cam_down(15)
#   13=cam_left(15) 14=cam_right(15) 15=sprint+attack
#   16=cam_up(5) 17=cam_down(5) 18=cam_left(5) 19=cam_right(5)
#   20=hotbar1 21=hotbar2 22=fwd+use 23=jump+attack 24=fwd+jump+attack
STAGE_PREFERRED_ACTIONS: Dict[str, List[int]] = {
    "approach":      [1, 3, 4, 8, 9, 10, 13, 14, 18, 19, 2],
    "navigate_deep": [12, 17, 6, 8, 1, 9, 24, 15, 2, 11],
    "aim":           [11, 12, 13, 14, 16, 17, 18, 19],
    "attack":        [6, 8, 15, 23, 24, 11, 12, 16, 17],
    "interact":      [7, 22, 12, 17, 11, 16, 1, 20, 21],
}

STAGE_MAX_STEPS: Dict[str, int] = {
    "approach":      80,
    "navigate_deep": 100,
    "aim":           40,
    "attack":        150,
    "interact":      60,
}

STAGE_CLIP_WEIGHT: Dict[str, float] = {
    "approach":      0.4,
    "navigate_deep": 0.4,
    "aim":           0.4,
    "attack":        0.2,
    "interact":      0.2,
}

APPROACH_CLIP_THRESHOLD      = 0.60
AIM_CLIP_THRESHOLD           = 0.65
NAVIGATE_DEEP_CLIP_THRESHOLD = 0.62


# ---------------------------------------------------------------------------
# Primitive reward wrappers
# ---------------------------------------------------------------------------

class PrimitiveRewardWrapper(gym.Wrapper if GYM_OK else object):
    """Base class for per-stage reward wrappers."""

    def __init__(self, env: MineRLObsWrapper,
                 clip_encoder=None, clip_weight: float = 0.3):
        if GYM_OK:
            super().__init__(env)
        else:
            self.env = env
        self._clip_enc    = clip_encoder
        self._clip_weight = clip_weight
        self._prev_obs: Optional[Dict[str, np.ndarray]] = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        self._prev_obs = obs
        return obs

    def step(self, action):
        obs, ext_rew, done, info = self.env.step(action)
        stage_r    = self._stage_reward(obs, self._prev_obs)
        early_done = self._is_done(obs, self._prev_obs)
        self._prev_obs = obs
        return obs, ext_rew + stage_r, done or early_done, info

    def _clip_score(self, obs: Dict[str, np.ndarray]) -> float:
        if (self._clip_enc is not None
                and "clip_goal" in obs and "rgb" in obs):
            return self._clip_enc.clip_reward(obs["rgb"], obs["clip_goal"])
        return 0.0

    def _stage_reward(self, obs, prev_obs) -> float:
        return 0.0

    def _is_done(self, obs, prev_obs) -> bool:
        return False


class ApproachRewardWrapper(PrimitiveRewardWrapper):
    """Navigate toward target. Done when CLIP > APPROACH_CLIP_THRESHOLD."""
    def _stage_reward(self, obs, prev_obs) -> float:
        return self._clip_weight * self._clip_score(obs) - 0.002

    def _is_done(self, obs, prev_obs) -> bool:
        return self._clip_score(obs) > APPROACH_CLIP_THRESHOLD


class NavigateDeepRewardWrapper(PrimitiveRewardWrapper):
    """Descend underground (diamond). Done when CLIP > NAVIGATE_DEEP_CLIP_THRESHOLD."""
    def _stage_reward(self, obs, prev_obs) -> float:
        return self._clip_weight * self._clip_score(obs) - 0.003

    def _is_done(self, obs, prev_obs) -> bool:
        return self._clip_score(obs) > NAVIGATE_DEEP_CLIP_THRESHOLD


class AimRewardWrapper(PrimitiveRewardWrapper):
    """Camera alignment. Done when CLIP > AIM_CLIP_THRESHOLD."""
    def _stage_reward(self, obs, prev_obs) -> float:
        return self._clip_weight * self._clip_score(obs) - 0.003

    def _is_done(self, obs, prev_obs) -> bool:
        return self._clip_score(obs) > AIM_CLIP_THRESHOLD


class _ItemRewardMixin:
    """Shared acquisition reward for attack and interact stages."""
    target_idx: int
    _prev_val:  float
    _clip_weight: float

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)  # type: ignore[misc]
        self._prev_val = float(obs["inventory"][self.target_idx])
        return obs

    def _stage_reward(self, obs, prev_obs) -> float:
        curr_val = float(obs["inventory"][self.target_idx])
        r = -0.005
        if curr_val > 0.5:
            r += 0.1
        if curr_val > self._prev_val + 0.5:
            r += 1.0
        self._prev_val = curr_val
        r += self._clip_weight * self._clip_score(obs)  # type: ignore[attr-defined]
        return r


class AttackRewardWrapper(_ItemRewardMixin, PrimitiveRewardWrapper):
    """Physical block-breaking reward (wood, stone, ores, diamond)."""
    def __init__(self, env, target_var_idx: int,
                 clip_encoder=None, clip_weight: float = 0.2):
        PrimitiveRewardWrapper.__init__(self, env,
                                         clip_encoder=clip_encoder,
                                         clip_weight=clip_weight)
        self.target_idx = target_var_idx
        self._prev_val  = 0.0


class InteractRewardWrapper(_ItemRewardMixin, PrimitiveRewardWrapper):
    """GUI right-click reward (furnace, crafting table)."""
    def __init__(self, env, target_var_idx: int,
                 clip_encoder=None, clip_weight: float = 0.2):
        PrimitiveRewardWrapper.__init__(self, env,
                                         clip_encoder=clip_encoder,
                                         clip_weight=clip_weight)
        self.target_idx = target_var_idx
        self._prev_val  = 0.0


# ── Wrapper factory ───────────────────────────────────────────────────────────

def make_primitive_wrapper(stage: str, env: MineRLObsWrapper,
                            target_var_idx: int,
                            clip_encoder=None,
                            clip_weight: Optional[float] = None):
    """Return the correct reward wrapper for the given stage."""
    if stage not in ALL_STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Valid: {ALL_STAGES}")
    w = clip_weight if clip_weight is not None else STAGE_CLIP_WEIGHT.get(stage, 0.3)
    if stage == "approach":
        return ApproachRewardWrapper(env, clip_encoder=clip_encoder, clip_weight=w)
    if stage == "navigate_deep":
        return NavigateDeepRewardWrapper(env, clip_encoder=clip_encoder, clip_weight=w)
    if stage == "aim":
        return AimRewardWrapper(env, clip_encoder=clip_encoder, clip_weight=w)
    if stage == "attack":
        return AttackRewardWrapper(env, target_var_idx=target_var_idx,
                                    clip_encoder=clip_encoder, clip_weight=w)
    if stage == "interact":
        return InteractRewardWrapper(env, target_var_idx=target_var_idx,
                                      clip_encoder=clip_encoder, clip_weight=w)


def validate_stage_for_var(var_name: str, stage: str) -> bool:
    """Return True iff stage is in the valid primitive chain for var_name."""
    return stage in SKILL_PRIMITIVE_CHAIN.get(var_name, [])


# ---------------------------------------------------------------------------
# PrimitivePPOOption — run one loaded PPO model for one stage
# ---------------------------------------------------------------------------

class PrimitivePPOOption:
    """Run a pre-trained SB3 PPO model for one primitive stage."""

    def __init__(self, model_or_path, max_steps: int, deterministic: bool = True):
        self.max_steps     = max_steps
        self.deterministic = deterministic
        self._model        = None
        if model_or_path is not None:
            self.load(model_or_path)

    def load(self, model_or_path):
        if isinstance(model_or_path, str):
            if not SB3_OK:
                raise ImportError("stable-baselines3 required")
            self._model = SB3PPO.load(model_or_path)
        else:
            self._model = model_or_path

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        if self._model is None:
            return int(np.random.randint(N_ACTIONS))
        action, _ = self._model.predict(obs, deterministic=self.deterministic)
        return int(action)

    def run_stage(self, env, evgs: EVGS, subgoal: Subgoal,
                  clip_goal: Optional[np.ndarray] = None,
                  ) -> Tuple[bool, int, Dict[str, np.ndarray]]:
        """Run for up to max_steps. Returns (success, steps, final_obs)."""
        if clip_goal is not None and hasattr(env, "set_goal_embedding"):
            env.set_goal_embedding(clip_goal)

        obs     = env.get_obs()
        steps   = 0
        success = False

        while steps < self.max_steps:
            action               = self.act(obs)
            next_obs, _, done, _ = env.step(action)

            x_curr = evgs.extract(obs)
            x_next = evgs.extract(next_obs)
            if EVGS.predicate_holds(x_curr, x_next, subgoal):
                success = True
                obs = next_obs
                break

            obs    = next_obs
            steps += 1
            if done:
                break

        return success, steps, obs


# ---------------------------------------------------------------------------
# CompositeSkillOption — chains skill-specific primitive sequence
# ---------------------------------------------------------------------------

class CompositeSkillOption(OptionPolicy):
    """
    Execute the primitive chain defined by SKILL_PRIMITIVE_CHAIN[var_name].

    At inference time the VLM can override any stage's CLIP goal via
    set_stage_goal(stage, text_or_embed).
    """

    def __init__(self,
                 subgoal: Subgoal,
                 cfg: OptionConfig,
                 var_name: str,
                 stages: Dict[str, Optional[PrimitivePPOOption]],
                 stage_order: List[str],
                 clip_encoder=None):
        super().__init__(subgoal, cfg)
        self.var_name  = var_name
        self._stages   = stages
        self._order    = stage_order
        self._clip_enc = clip_encoder
        self._clip_goals: Dict[str, Optional[np.ndarray]] = {s: None for s in ALL_STAGES}
        if clip_encoder is not None:
            self._precompute_clip_goals()

    def _precompute_clip_goals(self):
        for stage in self._order:
            text = (PRIMITIVE_TEXTS.get(stage, {})
                    .get(self.var_name, f"acquiring {self.var_name} in Minecraft"))
            self._clip_goals[stage] = self._clip_enc.encode_text(text)

    def set_stage_goal(self, stage: str, text_or_embed) -> None:
        """Override CLIP goal for a specific stage (VLM-provided instruction)."""
        if stage not in self._order:
            return
        if isinstance(text_or_embed, str):
            if self._clip_enc is None:
                return
            self._clip_goals[stage] = self._clip_enc.encode_text(text_or_embed)
        else:
            self._clip_goals[stage] = np.asarray(text_or_embed, dtype=np.float32)

    def act(self, obs) -> int:
        return 0  # not used; run() drives the episode

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        if not hasattr(env, "get_obs"):
            raise TypeError("CompositeSkillOption.run requires env with get_obs()")

        total_steps   = 0
        stage_results: Dict[str, Any] = {}

        for stage_name in self._order:
            primitive = self._stages.get(stage_name)
            if primitive is None:
                continue
            clip_goal = self._clip_goals.get(stage_name)
            success, n_steps, final_obs = primitive.run_stage(
                env, evgs, self.subgoal, clip_goal=clip_goal,
            )
            total_steps += n_steps
            stage_results[stage_name] = {"success": success, "steps": n_steps}

            if success:
                return {
                    "success":       True,
                    "steps":         total_steps,
                    "final_obs":     final_obs,
                    "stage_done":    stage_name,
                    "stage_results": stage_results,
                }

        return {
            "success":       False,
            "steps":         total_steps,
            "final_obs":     env.get_obs(),
            "stage_done":    self._order[-1] if self._order else None,
            "stage_results": stage_results,
        }

    @classmethod
    def from_paths(cls,
                   subgoal: Subgoal,
                   var_name: str,
                   model_dir: str = "models/minerl",
                   clip_encoder=None,
                   stage_max_steps: Optional[Dict[str, int]] = None,
                   deterministic: bool = True) -> "CompositeSkillOption":
        """Load primitives from disk using SKILL_PRIMITIVE_CHAIN[var_name]."""
        import os
        chain  = SKILL_PRIMITIVE_CHAIN.get(var_name, ["approach", "aim", "attack"])
        _steps = stage_max_steps or STAGE_MAX_STEPS

        stages: Dict[str, Optional[PrimitivePPOOption]] = {}
        for stage in chain:
            path = primitive_model_path(model_dir, var_name, stage)
            stages[stage] = (
                PrimitivePPOOption(path, max_steps=_steps.get(stage, 100),
                                   deterministic=deterministic)
                if os.path.exists(path) else None
            )

        total_max = sum(_steps.get(s, 100) for s in chain)
        cfg = OptionConfig(max_steps=total_max, terminate_on_success=True)
        return cls(subgoal, cfg, var_name, stages, chain, clip_encoder=clip_encoder)


# ---------------------------------------------------------------------------
# Path / status helpers
# ---------------------------------------------------------------------------

def primitive_model_path(model_dir: str, var_name: str, stage: str) -> str:
    import os
    return os.path.join(model_dir, f"{stage}_{var_name}.zip")


def load_composite_option(subgoal: Subgoal,
                           var_name: str,
                           model_dir: str = "models/minerl",
                           clip_encoder=None,
                           deterministic: bool = True) -> CompositeSkillOption:
    """Load all valid primitive models for var_name from model_dir."""
    return CompositeSkillOption.from_paths(
        subgoal, var_name, model_dir=model_dir,
        clip_encoder=clip_encoder, deterministic=deterministic,
    )


def print_primitive_status(var_names, model_dir: str = "models/minerl"):
    """Print which primitive models exist per skill."""
    import os
    print(f"\nPrimitive model status ({model_dir}):")
    for var in var_names:
        chain = SKILL_PRIMITIVE_CHAIN.get(var, [])
        marks = []
        for stage in chain:
            p    = primitive_model_path(model_dir, var, stage)
            mark = "\u2713" if os.path.exists(p) else " "
            marks.append(f"[{mark}]{stage}")
        print(f"  {var:<14} " + " \u2192 ".join(marks))
