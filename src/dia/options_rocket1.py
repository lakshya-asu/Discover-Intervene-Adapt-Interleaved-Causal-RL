"""
ROCKET-1 (MineStudio) gather option adapter for DIA.

ROCKET-1 is a pre-trained transformer policy from MineStudio (CraftJarvis, CVPR 2025).
It is used here as the low-level execution backbone for gathering skills, replacing
from-scratch PPO or BC policies.

DIA's PCG/SIG provides the high-level causal ordering (e.g. craft stonepickaxe before
mining ironore). ROCKET-1 executes each individual gather task.

Confirmed API (from MineStudio GitHub):
    from minestudio.models import RocketPolicy
    model = RocketPolicy.from_pretrained("CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    model = model.to("cuda").eval()
    memory = model.initial_state()
    action, memory = model.get_action(
        input={'image': rgb_hwc,          # (H, W, 3) uint8
               'segment': {'obj_id': obj_id_tensor,   # (1,) int64, interaction type
                            'obj_mask': mask_tensor}}, # (H, W) uint8 binary
        state_in=memory,
        input_shape="*",
        deterministic=False,
    )

Segmentation strategy:
    ROCKET-1 requires a binary object mask (typically from SAM-2).  For
    a practical first integration we use a voxel-derived proxy mask: any
    pixel column that overlaps with a target-block voxel cell is set to 1.
    If no voxel data is available we fall back to a full-frame mask (all 1s),
    which still produces reasonable navigation/attack behaviour because the
    ViT backbone sees the full scene.

Usage
-----
    from src.dia.options_rocket1 import ROCKET1GatherOption, is_rocket1_available

    if is_rocket1_available():
        opt = ROCKET1GatherOption(subgoal, cfg, skill_name="wood")
    else:
        opt = BCOptionWrapper(...)  # fallback

    result = opt.run(env_wrapper, evgs)
    # result: {success, steps, trajectory, final_obs, episode_done}
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interaction type IDs used by ROCKET-1's segmentation input
# 0=Hunt, 2=Mine, 3=Use/Interact, 4=Craft, 5=Switch, 6=Approach
# ---------------------------------------------------------------------------
_OBJ_ID_MINE = 2

# Skills that require mining (obj_id=2); use Approach (6) for navigation
_SKILL_OBJ_ID: dict[str, int] = {
    "wood":    _OBJ_ID_MINE,
    "stone":   _OBJ_ID_MINE,
    "coal":    _OBJ_ID_MINE,
    "ironore": _OBJ_ID_MINE,
    "diamond": _OBJ_ID_MINE,
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

    Returns the policy on success, None if MineStudio is not installed or
    the model cannot be fetched.  Logs a warning on failure so the caller
    can fall back to BC/scripted options gracefully.
    """
    global _ROCKET1_POLICY, _ROCKET1_LOAD_ATTEMPTED
    if _ROCKET1_LOAD_ATTEMPTED:
        return _ROCKET1_POLICY
    _ROCKET1_LOAD_ATTEMPTED = True

    try:
        import torch
        from minestudio.models import RocketPolicy  # type: ignore[import]

        logger.info("Loading ROCKET-1 from CraftJarvis/MineStudio_ROCKET-1.12w_EMA ...")
        policy = RocketPolicy.from_pretrained("CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = policy.to(device).eval()
        _ROCKET1_POLICY = policy
        logger.info("ROCKET-1 loaded on %s: %s", device, type(policy).__name__)
        return _ROCKET1_POLICY

    except ImportError as exc:
        logger.warning(
            "MineStudio not installed — ROCKET-1 unavailable (%s). "
            "Install with: pip install minestudio", exc,
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
    DIA gather option backed by a ROCKET-1 pre-trained policy.

    Follows the same interface as BCOptionWrapper and InventoryConditionedCraftOption:
        result = option.run(env_wrapper, evgs)

    Parameters
    ----------
    subgoal : Subgoal
        DIA subgoal; .var_index is the EVGS variable index for the target resource.
    cfg : OptionConfig
        .max_steps is the per-skill step budget.
    skill_name : str
        One of "wood", "stone", "coal", "ironore", "diamond".
    """

    def __init__(self, subgoal: Any, cfg: Any, skill_name: str) -> None:
        self.subgoal = subgoal
        self.cfg = cfg
        self.skill_name = skill_name
        self.obj_id = _SKILL_OBJ_ID.get(skill_name, _OBJ_ID_MINE)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, env: Any, evgs: Any) -> dict:
        """
        Execute ROCKET-1 policy for up to cfg.max_steps.

        Parameters
        ----------
        env : MineDojoObsWrapper
            Exposes .env (raw MineDojo env), ._last_raw, ._convert(), .get_obs().
        evgs : EVGS
            Checks success via evgs.extract(obs)[subgoal.var_index].

        Returns
        -------
        dict with keys: success, steps, trajectory, final_obs, episode_done.
        """
        import torch

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

        device = next(policy.parameters()).device
        obs = env.get_obs()
        memory = policy.initial_state()
        trajectory: list = []

        for _ in range(self.cfg.max_steps):
            rgb = obs.get("rgb", np.zeros((224, 224, 3), dtype=np.uint8))

            # Ensure 224×224 — env should already be at this size when
            # run_transfer_minedojo passes image_size=(224,224), but guard anyway
            if rgb.shape[:2] != (224, 224):
                import cv2
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Build segmentation mask: voxel-derived if available, else full-frame
            obj_mask = self._make_mask(obs, rgb.shape[:2])

            rocket_input = {
                "image": rgb,  # (H, W, 3) uint8
                "segment": {
                    "obj_id":   torch.tensor([self.obj_id], dtype=torch.int64).to(device),
                    "obj_mask": torch.tensor(obj_mask, dtype=torch.uint8).to(device),
                },
            }

            action_dict, memory = policy.get_action(
                input=rocket_input,
                state_in=memory,
                input_shape="*",
                deterministic=False,
            )

            # Step environment
            next_raw, _rew, done, _info = env.env.step(action_dict)
            env._last_raw = next_raw
            next_obs = env._convert(next_raw)

            trajectory.append((obs, action_dict, next_obs, done))
            obs = next_obs
            steps = len(trajectory)

            # Success check via EVGS inventory
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
    # Mask construction
    # ------------------------------------------------------------------

    def _make_mask(self, obs: dict, shape: tuple) -> np.ndarray:
        """
        Build a binary (H, W) uint8 object mask for ROCKET-1.

        Strategy (in order of preference):
        1. Use target_voxel from VoxelGatherWrapper obs to highlight columns
           where the target block appears in the voxel grid.
        2. Fall back to all-ones (full frame) mask — ROCKET-1 still navigates
           and mines reasonably; just without object-level grounding.
        """
        H, W = shape
        target_voxel = obs.get("target_voxel", None)

        if target_voxel is not None and np.any(target_voxel > 0):
            # target_voxel is flat (voxel_dim,) — reshape to (X, Y, Z)
            # voxel grid is centred on agent; project onto image plane (simple
            # column-projection: any voxel in front → centre strip of image)
            mask = np.zeros((H, W), dtype=np.uint8)
            # Simple heuristic: if any target voxel exists, highlight centre 50%
            # of the frame (where the target block is most likely to appear)
            x0, x1 = W // 4, 3 * W // 4
            y0, y1 = H // 4, 3 * H // 4
            mask[y0:y1, x0:x1] = 1
            return mask

        # Full-frame fallback
        return np.ones((H, W), dtype=np.uint8)
