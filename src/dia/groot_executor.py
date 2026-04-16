"""
GROOT skill executor for DIA+GROOT hybrid agent.

GrootPolicy (groot_one) is a goal-conditioned IL policy that takes a reference
video clip as goal specification.  The clip is encoded once per skill via
GrootPolicy.encode_video() and cached in policy.condition_cache.

Unlike ROCKET-1, GROOT does not require SAM-2 segmentation masks — the
reference video provides the visual goal signal directly.

Loaded from HuggingFace: CraftJarvis/MineStudio_GROOT.18w_EMA

Interface matches the ROCKET-1 runner's gather skill contract so it can be
used as a drop-in executor in the DIA planning loop.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill achievement helpers (mirrored from run_rocket1_minestudio.py)
# ---------------------------------------------------------------------------

_SKILL_ITEMS: Dict[str, List[str]] = {
    "wood":         ["log", "oak_log", "birch_log", "spruce_log",
                     "jungle_log", "acacia_log", "dark_oak_log"],
    "woodpickaxe":  ["wooden_pickaxe", "minecraft:wooden_pickaxe"],
    "stone":        ["cobblestone", "stone"],
    "coal":         ["coal"],
    "ironore":      ["iron_ore", "raw_iron"],
    "furnace":      ["furnace"],
    "stonepickaxe": ["stone_pickaxe"],
    "iron":         ["iron_ingot"],
    "ironpickaxe":  ["iron_pickaxe"],
    "diamond":      ["diamond"],
}


def _inv_has(info: Dict, item_names: List[str]) -> bool:
    """Return True if any item_name appears in the inventory with quantity >= 1."""
    item_set = {n.lower() for n in item_names}
    inv = info.get("inventory", {})
    if isinstance(inv, dict):
        names = inv.get("name", [])
        qtys  = inv.get("quantity", [])
        if hasattr(names, "__len__") and len(names) > 0:
            for n, q in zip(names, qtys):
                if str(n).strip().lower().rstrip("\x00") in item_set and int(q) > 0:
                    return True
        for v in inv.values():
            if isinstance(v, dict):
                if str(v.get("type", "")).lower() in item_set and int(v.get("quantity", 0)) > 0:
                    return True
    pickup = info.get("pickup", {})
    return any(pickup.get(name, 0) > 0 for name in item_names)


def _inv_count(info: Dict, item_names: List[str]) -> int:
    """Return total inventory quantity across all matching item_names."""
    item_set = {n.lower() for n in item_names}
    total = 0
    inv = info.get("inventory", {})
    if isinstance(inv, dict):
        names = inv.get("name", [])
        qtys  = inv.get("quantity", [])
        if hasattr(names, "__len__") and len(names) > 0:
            for n, q in zip(names, qtys):
                if str(n).strip().lower().rstrip("\x00") in item_set:
                    total += max(0, int(q))
        for v in inv.values():
            if isinstance(v, dict):
                if str(v.get("type", "")).lower() in item_set:
                    total += max(0, int(v.get("quantity", 0)))
    return total


def _skill_achieved(skill_name: str, info: Dict) -> bool:
    """Return True if the skill's target item is present in the agent's inventory."""
    return _inv_has(info, _SKILL_ITEMS.get(skill_name, [skill_name]))


# ---------------------------------------------------------------------------
# GrootExecutor
# ---------------------------------------------------------------------------

class GrootExecutor:
    """Wraps GrootPolicy for use inside the DIA planning loop.

    Memory lifecycle
    ----------------
    - ``reset()`` clears the recurrent state (call once per new episode).
    - ``run_skill()`` initialises memory on first call (after reset), then
      threads it across skill calls to preserve episodic context.
    - Callers receive the updated memory in the return tuple and may pass it
      back via ``self._memory`` assignment if they need cross-call continuity
      beyond what ``run_skill`` already provides.
    """

    def __init__(self, policy: Any, device: str = "cuda") -> None:
        self._policy = policy
        self._device = device
        self._memory: Optional[List[Any]] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        device: str = "cuda",
        use_ft_ckpt: bool = True,
    ) -> Optional["GrootExecutor"]:
        """Load GrootPolicy from HuggingFace.  Returns None if unavailable.

        Parameters
        ----------
        use_ft_ckpt:
            If True (default), load the fine-tuned checkpoint from
            ``data/groot_ft/groot_ft.pt`` when present.
            Set to False to force the original pretrained weights — useful
            when the fine-tuned model regresses (e.g. collapsed to dominant
            action) and you want the pristine visual goal-following behaviour.
        """
        try:
            from minestudio.models.groot_one.body import load_groot_policy
        except ImportError as exc:
            logger.warning("GrootPolicy unavailable (import error): %s", exc)
            return None

        try:
            logger.info("Loading GrootPolicy (CraftJarvis/MineStudio_GROOT.18w_EMA) …")
            policy = load_groot_policy()
            policy = policy.to(device)

            # Optionally load fine-tuned checkpoint
            ft_ckpt = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../data/groot_ft/groot_ft.pt",
            )
            if use_ft_ckpt and os.path.isfile(ft_ckpt):
                logger.info("Loading fine-tuned GROOT from %s", ft_ckpt)
                policy.load_state_dict(
                    torch.load(ft_ckpt, map_location=device, weights_only=True)
                )
            elif not use_ft_ckpt:
                logger.info("use_ft_ckpt=False — using original pretrained weights.")
            else:
                logger.info("No fine-tuned checkpoint found — using pretrained weights.")

            policy = policy.eval()
            logger.info("GrootPolicy loaded on %s.", device)
            return cls(policy, device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GrootPolicy load failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset recurrent memory.  Call at the start of each new episode."""
        self._memory = None
        logger.debug("GrootExecutor memory reset.")

    # ------------------------------------------------------------------
    # Navigation primers
    # ------------------------------------------------------------------

    # Sweep format: (pitch_delta, yaw_delta, force_attack, force_forward)
    # force_attack/force_forward=1 → override GROOT's button output with 1
    # (used during the underground dig phase where GROOT outputs NOOP because
    # the bright-surface observation doesn't match dark-cave reference clips)

    # Surface primer: horizontal yaw sweep so the agent faces a tree/stone outcrop.
    # GROOT controls buttons throughout — it reliably produces attack+forward at
    # trunk level once the camera is aimed at a tree or stone face.
    _SURFACE_SWEEP: List[Tuple[float, float, int, int]] = (
        [(0.0,  15.0, 0, 0)] * 12   # turn right 180°
        + [(0.0, -15.0, 0, 0)] * 12 # sweep back (and left 180°)
        + [(0.0,  15.0, 0, 0)] * 6  # re-centre
        + [(5.0,  0.0,  0, 0)] * 4  # pitch slightly down (trunk level)
        + [(-5.0, 0.0,  0, 0)] * 4  # restore
    )

    # Underground primer: two-phase subgoal sequence.
    #
    # Subgoal 1 — Reach underground (steps 0-152):
    #   Pitch camera to 45° and FORCE attack+forward for 150 steps, digging a
    #   diagonal staircase.  GROOT's buttons are ignored here because the surface
    #   visual (bright daylight) doesn't match the dark-cave BC reference clips.
    #   150 uninterrupted steps = deepest possible descent, crossing more ore veins.
    #   (Pausing for item collection reduced depth and caused ironore regression.)
    #
    # Subgoal 2 — Visual scan for ore (steps 153-199):
    #   Camera pitches back toward horizontal and sweeps left/right.  GROOT
    #   controls attack — it visually identifies ore seams and attacks them.
    #   This is the it5 behaviour: GROOT's visual goal-following drives attack
    #   once it is at cave depth and can see ore-matching blocks.
    _UNDERGROUND_SWEEP: List[Tuple[float, float, int, int]] = (
        [(15.0, 0.0,  0, 0)] * 3    # pitch down to +45°  (GROOT buttons)
        + [(0.0, 0.0,  1, 1)] * 150 # FORCED dig staircase at 45° pitch
        + [(0.0, 15.0, 0, 0)] * 8   # sweep right (GROOT controls attack)
        + [(0.0, -15.0,0, 0)] * 8   # sweep left  (GROOT controls attack)
        + [(0.0, 15.0, 0, 0)] * 4   # sweep right again
        + [(-5.0, 0.0, 0, 0)] * 5   # restore pitch toward horizontal
        + [(0.0, 0.0,  0, 0)] * 22  # face forward (GROOT controls attack)
    )

    def _run_primer(
        self,
        sweep: List[Tuple[float, float, int, int]],
        label: str,
        clip_path: str,
        env: Any,
        obs: Dict,
        info: Dict,
        skill_name: str,
        n_steps: int,
    ) -> Tuple[Dict, Dict, int]:
        """Execute a camera-override primer for *n_steps* steps.

        Each sweep entry is (pitch_delta, yaw_delta, force_attack, force_forward).
        Camera is always overridden by the program.  When force_attack/force_forward
        are 1, those buttons are set unconditionally (overriding GROOT's prediction).
        Returns early if the skill is achieved or the episode ends.
        """
        if self._memory is None:
            self._memory = self._policy.initial_state()

        # Extend sweep with no-op padding (no button overrides) if needed
        padding = [(0.0, 0.0, 0, 0)] * max(0, n_steps - len(sweep))
        program = (sweep + padding)[:n_steps]
        steps_done = 0

        for pitch_d, yaw_d, force_atk, force_fwd in program:
            try:
                action_dict, self._memory = self._policy.get_action(
                    input={"image": obs["image"], "ref_video_path": clip_path},
                    state_in=self._memory,
                    input_shape="*",
                    deterministic=False,
                )
                # Camera always overridden by primer program
                if "camera" in action_dict and hasattr(action_dict["camera"], "__len__"):
                    action_dict["camera"] = np.array([pitch_d, yaw_d], dtype=np.float32)
                # Forced button overrides for dig phase
                if force_atk:
                    action_dict["attack"]  = 1
                    action_dict["forward"] = 1
                    if "sprint" in action_dict:
                        action_dict["sprint"] = 1
                if force_fwd and not force_atk:
                    action_dict["forward"] = 1

                obs, _r, terminated, truncated, info = env.step(action_dict)
                steps_done += 1
                if _skill_achieved(skill_name, info):
                    logger.info("[groot] %s primer: skill=%s achieved at step %d.",
                                label, skill_name, steps_done)
                    break
                if terminated or truncated:
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning("[groot] %s primer error at step %d: %s",
                               label, steps_done, exc)
                break

        logger.info("[groot] %s primer done (%d steps).", label, steps_done)
        return obs, info, steps_done

    # ------------------------------------------------------------------
    # Skill execution
    # ------------------------------------------------------------------

    def run_skill(
        self,
        skill_name: str,
        clip_path: str,
        env: Any,
        obs: Dict,
        info: Dict,
        max_steps: int = 2000,
        primer_steps: int = 0,
        primer_type: str = "surface",
        min_qty: int = 1,
    ) -> Tuple[bool, int, Dict, Dict, Any]:
        """Run GROOT for one skill using *clip_path* as the goal reference.

        Parameters
        ----------
        skill_name:
            Human-readable skill identifier.
        clip_path:
            Absolute path to a ``.mp4`` reference clip.
        env:
            MineStudio / MineRL environment instance.
        obs:
            Current observation dict (must contain ``"image"`` key).
        info:
            Current info dict from the environment.
        max_steps:
            Maximum env steps to execute before giving up.
        primer_steps:
            If > 0, run a navigation primer for this many steps before
            handing full control to GROOT.
        primer_type:
            ``"surface"`` — horizontal yaw sweep (wood, stone).
            ``"underground"`` — pitch-down staircase descent then wall sweep
            (coal, ironore, diamond).  The agent digs diagonally for ~150
            steps to reach ore depth, then scans walls for the ore seam.
        min_qty:
            Keep farming until inventory holds at least this many of the
            target item.  Use > 1 for gather skills (e.g. stone needs 11
            for furnace + stonepickaxe) so the agent doesn't exit after
            the first pickup and then have to dig a new hole later.

        Returns
        -------
        (success, steps_taken, obs, info, memory)
        """
        # Initialise memory on first call after reset
        if self._memory is None:
            self._memory = self._policy.initial_state()

        steps = 0
        success = False
        skill_items = _SKILL_ITEMS.get(skill_name, [skill_name])

        # Optional navigation primer
        if primer_steps > 0:
            sweep = (self._UNDERGROUND_SWEEP if primer_type == "underground"
                     else self._SURFACE_SWEEP)
            obs, info, primer_done = self._run_primer(
                sweep=sweep,
                label=primer_type,
                clip_path=clip_path,
                env=env,
                obs=obs,
                info=info,
                skill_name=skill_name,
                n_steps=primer_steps,
            )
            steps += primer_done
            # Check min_qty even during primer exit
            if _inv_count(info, skill_items) >= min_qty:
                logger.info("[groot] skill=%s qty>=%d achieved during %s primer at step %d.",
                            skill_name, min_qty, primer_type, steps)
                return True, steps, obs, info, self._memory

        try:
            for _s in range(1, max_steps - steps + 1):
                steps += 1
                action_dict, self._memory = self._policy.get_action(
                    input={
                        "image":         obs["image"],
                        "ref_video_path": clip_path,
                    },
                    state_in=self._memory,
                    input_shape="*",
                    deterministic=False,
                )

                obs, _reward, terminated, truncated, info = env.step(action_dict)

                if steps % 100 == 0:
                    qty = _inv_count(info, skill_items)
                    logger.info(
                        "[groot] skill=%s  step=%d/%d  qty=%d/%d",
                        skill_name, steps, max_steps, qty, min_qty,
                    )

                qty = _inv_count(info, skill_items)
                if qty >= min_qty:
                    success = True
                    logger.info(
                        "[groot] skill=%s qty=%d/%d reached at step %d.",
                        skill_name, qty, min_qty, steps,
                    )
                    break

                if terminated or truncated:
                    logger.info(
                        "[groot] episode ended (terminated=%s truncated=%s) "
                        "at step %d for skill=%s.",
                        terminated, truncated, steps, skill_name,
                    )
                    break

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[groot] error during skill=%s at step %d: %s",
                skill_name, steps, exc,
            )

        return success, steps, obs, info, self._memory
