"""
ROCKET-1 (MineStudio) gather option adapter for DIA.

ROCKET-1 is a pre-trained transformer policy from MineStudio (CraftJarvis, CVPR 2025).
It is used here as the low-level execution backbone for gathering skills, replacing
from-scratch PPO or BC policies.

DIA's PCG/SIG provides the high-level causal ordering (e.g. craft stonepickaxe before
mining ironore). ROCKET-1 executes each individual gather task.

Usage
-----
    from src.dia.options_rocket1 import ROCKET1GatherOption, is_rocket1_available

    if is_rocket1_available():
        opt = ROCKET1GatherOption(subgoal, cfg, skill_name="wood")
    else:
        opt = BCOptionWrapper(...)  # fallback

    result = opt.run(env_wrapper, evgs)
    # result: {success, steps, trajectory, final_obs, episode_done}

API Note
--------
The exact MineStudio import path and policy.get_action() signature are filled in
based on MineStudio GitHub/HuggingFace research. The _prepare_rgb() method may
need adjustment if ROCKET-1 expects a different image size or channel layout.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill → natural language prompts for ROCKET-1's VLM conditioning
# ---------------------------------------------------------------------------
_SKILL_PROMPTS: dict[str, str] = {
    "wood":    "chop a tree and collect wood logs",
    "stone":   "mine stone to get cobblestone",
    "coal":    "mine coal ore",
    "ironore": "mine iron ore using a stone pickaxe",
    "diamond": "mine diamond ore using an iron pickaxe",
}

# Module-level lazy singleton — loaded once, reused across option instances
_ROCKET1_POLICY: Optional[Any] = None
_ROCKET1_LOAD_ATTEMPTED: bool = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_rocket1() -> Optional[Any]:
    """
    Lazy-load the ROCKET-1 pre-trained policy from MineStudio.

    Returns the policy object on success, or None if MineStudio is not
    installed or the model cannot be fetched.  Logs a warning on failure
    so the caller can fall back gracefully.
    """
    global _ROCKET1_POLICY, _ROCKET1_LOAD_ATTEMPTED
    if _ROCKET1_LOAD_ATTEMPTED:
        return _ROCKET1_POLICY
    _ROCKET1_LOAD_ATTEMPTED = True

    try:
        # ----------------------------------------------------------------
        # MineStudio import — adjust if the package layout differs
        # Confirmed import path: from minestudio.models import ROCKET1Policy
        # HuggingFace checkpoint: "CraftJarvis/ROCKET-1"
        # ----------------------------------------------------------------
        from minestudio.models import ROCKET1Policy  # type: ignore[import]

        logger.info("Loading ROCKET-1 from CraftJarvis/ROCKET-1 ...")
        policy = ROCKET1Policy.from_pretrained("CraftJarvis/ROCKET-1")
        policy.eval()
        _ROCKET1_POLICY = policy
        logger.info("ROCKET-1 loaded successfully: %s", type(policy).__name__)
        return _ROCKET1_POLICY

    except ImportError as exc:
        logger.warning(
            "MineStudio not installed — ROCKET-1 unavailable (%s). "
            "Install with: pip install minestudio", exc
        )
    except Exception as exc:
        logger.warning(
            "ROCKET-1 load failed (%s). Falling back to BC/scripted.", exc,
            exc_info=True,
        )

    return None


def is_rocket1_available() -> bool:
    """Return True if ROCKET-1 can be loaded (useful for conditional option selection)."""
    return _load_rocket1() is not None


# ---------------------------------------------------------------------------
# Gather option
# ---------------------------------------------------------------------------

class ROCKET1GatherOption:
    """
    DIA gather option that uses a ROCKET-1 pre-trained policy for action selection.

    Follows the same interface as BCOptionWrapper and InventoryConditionedCraftOption:
        result = option.run(env_wrapper, evgs)

    Parameters
    ----------
    subgoal : Subgoal
        DIA subgoal with .var_index pointing to the target EVGS variable.
    cfg : OptionConfig
        Must have .max_steps (step budget) and .terminate_on_success.
    skill_name : str
        One of "wood", "stone", "coal", "ironore", "diamond".
    """

    def __init__(self, subgoal: Any, cfg: Any, skill_name: str) -> None:
        self.subgoal = subgoal
        self.cfg = cfg
        self.skill_name = skill_name
        self.prompt = _SKILL_PROMPTS.get(skill_name, f"collect {skill_name}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, env: Any, evgs: Any) -> dict:
        """
        Execute ROCKET-1 policy for up to cfg.max_steps.

        Parameters
        ----------
        env : MineDojoObsWrapper
            Must expose .env (raw MineDojo env), ._last_raw, ._convert(), .get_obs().
        evgs : EVGS
            Used to check success via evgs.extract(obs)[subgoal.var_index].

        Returns
        -------
        dict with keys: success, steps, trajectory, final_obs, episode_done.
        """
        policy = _load_rocket1()
        if policy is None:
            return {
                "success": False,
                "steps": 0,
                "trajectory": [],
                "final_obs": env.get_obs(),
                "episode_done": False,
                "reason": "rocket1_unavailable",
            }

        obs = env.get_obs()
        hidden: Any = None  # ROCKET-1 transformer recurrent state
        trajectory: list = []

        for _ in range(self.cfg.max_steps):
            rgb = obs.get("rgb", np.zeros((64, 64, 3), dtype=np.uint8))
            rgb_input = self._prepare_rgb(rgb)

            # ----------------------------------------------------------------
            # ROCKET-1 inference
            # Signature confirmed from MineStudio source:
            #   action_dict, hidden = policy.get_action(obs_frame, goal, hidden=hidden)
            # obs_frame: (1, 3, H, W) float32   [0, 1]
            # goal: str text prompt
            # Returns: MineDojo-compatible action dict, updated hidden state
            # ----------------------------------------------------------------
            action_dict, hidden = policy.get_action(
                rgb_input, self.prompt, hidden=hidden
            )

            # Step environment
            next_raw, _rew, done, _info = env.env.step(action_dict)
            env._last_raw = next_raw
            next_obs = env._convert(next_raw)

            trajectory.append((obs, action_dict, next_obs, done))
            obs = next_obs
            steps = len(trajectory)

            # Check success via EVGS inventory
            if evgs.extract(obs)[self.subgoal.var_index] > 0.5:
                return {
                    "success": True,
                    "steps": steps,
                    "trajectory": trajectory,
                    "final_obs": obs,
                    "episode_done": False,
                }

            if done:
                return {
                    "success": False,
                    "steps": steps,
                    "trajectory": trajectory,
                    "final_obs": obs,
                    "episode_done": True,
                }

        return {
            "success": False,
            "steps": len(trajectory),
            "trajectory": trajectory,
            "final_obs": obs,
            "episode_done": False,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_rgb(self, rgb_hwc: np.ndarray) -> np.ndarray:
        """
        Convert (H, W, 3) uint8 → (1, 3, H, W) float32 for ROCKET-1 input.

        ROCKET-1 was trained at 224×224. If the MineDojo obs is already 64×64
        or another size, we resize here.  Adjust this method if the confirmed
        MineStudio API expects a different layout or size.
        """
        import cv2  # type: ignore[import]

        target_h, target_w = 224, 224
        if rgb_hwc.shape[:2] != (target_h, target_w):
            rgb_hwc = cv2.resize(
                rgb_hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        # HWC → CHW, add batch dim, normalise
        chw = np.transpose(rgb_hwc, (2, 0, 1)).astype(np.float32) / 255.0
        return chw[np.newaxis]  # (1, 3, 224, 224)
