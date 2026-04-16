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
# Canonical mapping from MineStudio tutorials/inference/evaluate_rocket/utils.py:
#   SEGMENT_MAPPING = {"Hunt":0, "Use":3, "Mine":2, "Interact":3,
#                       "Craft":4, "Switch":5, "Approach":6, "None":-1}
# ---------------------------------------------------------------------------
_OBJ_ID_MINE     = 2   # "Mine"     — break blocks
_OBJ_ID_APPROACH = 6   # "Approach" — navigate toward target

# When target IS visible (SAM-2 tracking): Mine (2) + SAM-2 mask
# When target NOT visible (searching):     Approach (6) + zero mask
# This matches how the official MineStudio demo uses ROCKET-1.

# Module-level lazy singleton — loaded once, reused across option instances
_ROCKET1_POLICY: Optional[Any] = None
_ROCKET1_LOAD_ATTEMPTED: bool = False

# ---------------------------------------------------------------------------
# Action conversion: VPT CameraHierarchical → MineDojo named dict
# ROCKET-1 outputs {'buttons': int (0-8640), 'camera': int (0-120)} in the
# VPT CameraHierarchical action space.  MineDojo env.step() expects individual
# named keys ('forward', 'attack', 'camera', ...).  This mirrors what
# MineStudio's simulator.entry.agent_action_to_env_action() does.
# ---------------------------------------------------------------------------

_ACTION_MAPPER: Optional[Any] = None
_ACTION_TRANSFORMER: Optional[Any] = None


def _get_action_converters():
    """Lazy-init action mapper and transformer (avoid import cost at module load)."""
    global _ACTION_MAPPER, _ACTION_TRANSFORMER
    if _ACTION_MAPPER is None:
        from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
        from minestudio.utils.vpt_lib.actions import ActionTransformer
        _ACTION_MAPPER = CameraHierarchicalMapping(n_camera_bins=11)
        _ACTION_TRANSFORMER = ActionTransformer(camera_maxval=10, camera_binsize=2)
    return _ACTION_MAPPER, _ACTION_TRANSFORMER


def _vpt_to_minedojo(action: dict) -> np.ndarray:
    """Convert ROCKET-1 VPT action to MineDojo 8-dim MultiDiscrete array.

    MineDojo NNActionSpaceWrapper dims (cam_interval=15°):
      [fb, lr, jss, pitch_bin, yaw_bin, fn, craft_arg, slot]
      fb:   0=noop, 1=forward, 2=back
      lr:   0=noop, 1=left,    2=right
      jss:  0=noop, 1=jump,    2=sneak,  3=sprint
      pitch_bin / yaw_bin: 0-24, noop=12 (bin*15-180=degrees)
      fn:   0=noop, 1=use, 2=drop, 3=attack, 4=craft, ...

    Pipeline:
      VPT {'buttons': 0-8640, 'camera': 0-120}
      → CameraHierarchicalMapping.to_factored  →  buttons (20,), camera (2,)
      → ActionTransformer.numpy_to_dict         →  named dict + continuous camera
      → map to MineDojo 8-dim array
    """
    mapper, transformer = _get_action_converters()
    ac = {
        "buttons": action["buttons"].cpu().numpy() if hasattr(action["buttons"], "cpu")
                   else np.asarray(action["buttons"]),
        "camera":  action["camera"].cpu().numpy()  if hasattr(action["camera"],  "cpu")
                   else np.asarray(action["camera"]),
    }
    factored = mapper.to_factored(ac)        # buttons (20,), camera (2,)
    named    = transformer.numpy_to_dict(factored)  # named keys + continuous camera

    arr = np.array([0, 0, 0, 12, 12, 0, 0, 0], dtype=np.int64)  # noop

    # dim 0: forward/back
    fwd = int(named.get("forward", 0))
    bck = int(named.get("back", 0))
    if fwd and not bck:
        arr[0] = 1
    elif bck and not fwd:
        arr[0] = 2

    # dim 1: left/right
    lft = int(named.get("left", 0))
    rgt = int(named.get("right", 0))
    if lft and not rgt:
        arr[1] = 1
    elif rgt and not lft:
        arr[1] = 2

    # dim 2: jump/sneak/sprint
    if   int(named.get("jump",   0)): arr[2] = 1
    elif int(named.get("sprint", 0)): arr[2] = 3
    elif int(named.get("sneak",  0)): arr[2] = 2

    # dims 3-4: camera (VPT gives ±10° continuous; MineDojo bin = (deg+180)/15, noop=12)
    cam = named.get("camera", np.zeros(2))
    if hasattr(cam, "__len__") and len(cam) >= 2:
        arr[3] = max(0, min(24, int(round((float(cam[0]) + 180) / 15))))
        arr[4] = max(0, min(24, int(round((float(cam[1]) + 180) / 15))))

    # dim 5: fn (attack takes priority over use)
    if   int(named.get("attack", 0)): arr[5] = 3
    elif int(named.get("use",    0)): arr[5] = 1

    return arr


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
        # SAM-2 tracker — built lazily on first run(), None if unavailable
        self._sam2: Optional[Any] = None
        self._sam2_tried: bool = False

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
        # Keep the raw MineDojo obs around for voxel-based mask generation.
        # MinedojoObsWrapper stores it as _last_obs; some wrappers use _last_raw.
        raw_obs = getattr(env, '_last_obs', None) or getattr(env, '_last_raw', None)

        # ── Lazy-build SAM-2 tracker (once per option instance) ───────────
        if not self._sam2_tried:
            self._sam2_tried = True
            try:
                from dia.sam2_tracker import SAM2Tracker
                _dev = "cuda" if str(device) != "cpu" else "cpu"
                self._sam2 = SAM2Tracker.build(device=_dev)
                if self._sam2 is not None:
                    logger.info("SAM-2 tracker ready for skill '%s'", self.skill_name)
                else:
                    logger.info("SAM-2 unavailable — using voxel/full-frame mask")
            except Exception as exc:
                logger.warning("SAM-2 build failed (%s) — using voxel/full-frame mask", exc)
        if self._sam2 is not None:
            self._sam2.reset()  # fresh tracking session for each run() call

        # Use None for first call — initial_state() returns squeezed tensors that
        # cause a dim mismatch when passed back through get_action("*")'s single
        # unsqueeze. Model creates correct internal state when state_in=None.
        memory = None
        trajectory: list = []

        for _ in range(self.cfg.max_steps):
            rgb = self._get_rgb_224(obs)
            H, W = rgb.shape[:2]

            # ── Two-phase interaction type selection ──────────────────────────
            # Phase 1 (searching): obj_id=Approach(6), mask=zeros
            #   → ROCKET-1 navigates toward the target area
            # Phase 2 (target visible): obj_id=Mine(2), mask=SAM-2
            #   → ROCKET-1 aims at and breaks the specific block
            # This matches the official MineStudio demo (session.segment_type="Approach"
            # with zeros mask initially; switches to Mine when SAM-2 is initialised).
            # Get the best seed point for SAM-2.
            # Priority: voxel projection (precise) → image center (forest fallback).
            # ROCKET-1 was trained with always-present SAM-2 masks (from Molmo VLM
            # or human click). Without a seed we lose the spatial grounding entirely.
            # In a forest biome the image center is almost always a tree log or leaf,
            # making center a reasonable fallback when voxels don't reach the tree.
            seed_xy = self._voxel_seed_point(raw_obs, H, W)
            if seed_xy is None:
                seed_xy = (W // 2, H // 2)  # center fallback (works well in forests)

            if self._sam2 is not None:
                obj_mask = self._sam2.update(rgb, seed_xy=seed_xy)
            else:
                obj_mask = self._make_mask(obs, (H, W), raw_obs=raw_obs)

            # When SAM-2 has a valid mask → Mine (2); when mask collapsed/zeros → Mine
            # still (ROCKET-1 needs Mine mode to chop blocks even if mask is imperfect).
            active_obj_id = _OBJ_ID_MINE

            rocket_input = {
                "image": rgb,  # (H, W, 3) uint8
                "segment": {
                    # obj_id must be a scalar (0-D) so that _batchify's 2x unsqueeze
                    # produces (1, 1) — 2D — making interaction() output (1, 1, hiddim)
                    # 3D.  A 1-D [val] would become (1,1,1) → (1,1,1,hiddim) 4D,
                    # which then breaks the KV-cache cat in xf.py update_state.
                    "obj_id":   torch.tensor(active_obj_id, dtype=torch.int64).to(device),
                    "obj_mask": torch.tensor(obj_mask, dtype=torch.uint8).to(device),
                },
            }

            action_dict, memory = policy.get_action(
                input=rocket_input,
                state_in=memory,
                input_shape="*",
                deterministic=False,
            )

            # Convert VPT {'buttons', 'camera'} → MineDojo named action dict
            minedojo_action = _vpt_to_minedojo(action_dict)

            # Step environment
            next_raw, _rew, done, _info = env.env.step(minedojo_action)
            # Update cached raw obs on the wrapper (both attribute names used across wrappers)
            env._last_raw = next_raw
            if hasattr(env, '_last_obs'):
                env._last_obs = next_raw
            next_obs = env._convert(next_raw)
            raw_obs = next_raw  # update for next iteration's mask

            trajectory.append((obs, action_dict, next_obs, done))
            obs = next_obs
            steps = len(trajectory)

            # Success check: obs["inventory"] is the EVGS binary vector (9,) float32
            # stored by MinedojoObsWrapper._convert().  Direct indexing is faster
            # and avoids re-extracting from the raw MineDojo inventory format.
            inv_vec = obs.get("inventory") if isinstance(obs, dict) else None
            got_it = (inv_vec is not None
                      and inv_vec[self.subgoal.var_index] > 0.5)
            if got_it:
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
    # Voxel seed point for SAM-2 initialisation
    # ------------------------------------------------------------------

    def _voxel_seed_point(
        self, raw_obs: Any, H: int, W: int
    ) -> Optional[tuple]:
        """
        Return (cx, cy) image pixel coords of the most visible target voxel,
        or None if no target voxel is in front of the agent.

        Uses the same perspective projection as _voxel_to_mask but returns
        only the single best point (highest fwd_dist = most likely visible).
        """
        import math
        if raw_obs is None or not isinstance(raw_obs, dict):
            return None
        try:
            from dia.evgs_minedojo import SKILL_TARGETS
        except ImportError:
            return None

        targets = {t.lower() for t in SKILL_TARGETS.get(self.skill_name, [])}
        if not targets:
            return None

        voxel_obs = raw_obs.get("voxels", None)
        if voxel_obs is None:
            return None

        try:
            if isinstance(voxel_obs, dict):
                block_names = voxel_obs.get("block_name", None)
            elif getattr(voxel_obs, "dtype", None) and getattr(voxel_obs.dtype, "names", None):
                block_names = (voxel_obs["block_name"]
                               if "block_name" in voxel_obs.dtype.names else None)
            else:
                block_names = None
            if block_names is None:
                return None

            bn_arr = np.asarray(block_names)
            if bn_arr.ndim < 3:
                return None

            X, Y, Z = bn_arr.shape
            cx_g, cy_g, cz_g = X // 2, Y // 2, Z // 2

            loc = raw_obs.get("location_stats", {}) or {}
            def _flt(d, k, dv=0.0):
                v = d.get(k, dv)
                if v is None: return dv
                try: return float(np.asarray(v).ravel()[0])
                except: return dv

            ya = math.radians(_flt(loc, "yaw",   0.0))
            pa = math.radians(_flt(loc, "pitch", 0.0))
            fw = np.array([-math.sin(ya)*math.cos(pa), -math.sin(pa), math.cos(ya)*math.cos(pa)])
            world_up = np.array([0.0, 1.0, 0.0])
            rt = np.cross(world_up, fw)
            rt_norm = np.linalg.norm(rt)
            rt = rt / rt_norm if rt_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
            up = np.cross(fw, rt)

            best_fwd, best_xy = -1.0, None
            for xi in range(X):
                for yi in range(Y):
                    for zi in range(Z):
                        name = str(bn_arr[xi, yi, zi]).lower().strip().rstrip("\x00")
                        if name not in targets:
                            continue
                        world_off = np.array([float(xi-cx_g), float(yi-cy_g), float(zi-cz_g)])
                        fwd_dist = float(np.dot(fw, world_off))
                        if fwd_dist <= 0.1:
                            continue
                        if fwd_dist > best_fwd:
                            best_fwd = fwd_dist
                            x_cam = float(np.dot(rt, world_off))
                            y_cam = float(np.dot(up, world_off))
                            FOCAL = 1.0
                            u = (x_cam / (fwd_dist * FOCAL)) * 0.5 + 0.5
                            v = (-y_cam / (fwd_dist * FOCAL)) * 0.5 + 0.5
                            img_x = int(np.clip(u * W, 0, W - 1))
                            img_y = int(np.clip(v * H, 0, H - 1))
                            best_xy = (img_x, img_y)

            return best_xy  # None if no in-front target

        except Exception as exc:
            logger.debug("_voxel_seed_point error (%s)", exc)
            return None

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    def _make_mask(self, obs: dict, shape: tuple, raw_obs: Any = None) -> np.ndarray:
        """
        Build a binary (H, W) uint8 object mask for ROCKET-1.

        Strategy (in order of preference):
        1. Voxel data from raw MineDojo obs (raw_obs["voxels"]["block_name"]).
           Project 3-D voxel positions to image columns/rows using a simple
           perspective heuristic.  MineDojo voxels are a centred grid; the
           agent is at the centre voxel.  Axes: X=east, Y=up, Z=south.
           Because Minecraft camera looks in the –Z or +Z direction depending
           on yaw, we use a simplified mapping:
             • forward voxels  → centre columns
             • left/right      → left/right columns
             • high Y          → upper rows, low Y → lower rows
        2. pre-processed target_voxel in obs (legacy VoxelGatherWrapper path).
        3. Full-frame fallback — ROCKET-1 still navigates/mines, just without
           object-level spatial grounding.
        """
        H, W = shape

        # ── 1. Voxel data from raw MineDojo obs ───────────────────────────────
        if raw_obs is not None and isinstance(raw_obs, dict):
            mask = self._voxel_to_mask(raw_obs, H, W)
            if mask is not None:
                return mask

        # ── 2. Legacy pre-processed target_voxel (VoxelGatherWrapper) ─────────
        target_voxel = obs.get("target_voxel", None) if isinstance(obs, dict) else None
        if target_voxel is not None and np.any(target_voxel > 0):
            mask = np.zeros((H, W), dtype=np.uint8)
            x0, x1 = W // 4, 3 * W // 4
            y0, y1 = H // 4, 3 * H // 4
            mask[y0:y1, x0:x1] = 1
            return mask

        # ── 3. Full-frame fallback ─────────────────────────────────────────────
        return np.ones((H, W), dtype=np.uint8)

    def _voxel_to_mask(self, raw_obs: dict, H: int, W: int) -> Optional[np.ndarray]:
        """
        Project MineDojo voxel data to a (H, W) uint8 mask using a perspective
        camera model derived from the agent's pitch/yaw.

        MineDojo voxels are a 5×5×5 grid (when use_voxel=True with xmin/xmax=±2)
        in world coordinates: +X=east, +Y=up, +Z=south.  The agent is at the
        centre cell (2, 2, 2).

        We rotate each target-block's world-relative offset into camera space
        using the agent's pitch and yaw from location_stats, then apply a
        pinhole-camera perspective division to obtain (u, v) image coordinates.

        Minecraft yaw convention (degrees):
          0   → facing south (+Z)
          90  → facing west  (-X)
          180 → facing north (-Z)
          -90 → facing east  (+X)

        Minecraft pitch convention (degrees):
          0    → horizontal
          -90  → looking up   (-Y)
          +90  → looking down (+Y)

        Returns None if voxel data is absent or no in-front target block found.
        """
        import math
        try:
            from dia.evgs_minedojo import SKILL_TARGETS
        except ImportError:
            return None

        targets = {t.lower() for t in SKILL_TARGETS.get(self.skill_name, [])}
        if not targets:
            return None

        voxel_obs = raw_obs.get("voxels", None)
        if voxel_obs is None:
            return None

        try:
            # ── Extract block_name array ──────────────────────────────────────
            if isinstance(voxel_obs, dict):
                block_names = voxel_obs.get("block_name", None)
            elif hasattr(voxel_obs, "dtype") and getattr(voxel_obs.dtype, "names", None):
                block_names = (voxel_obs["block_name"]
                               if "block_name" in voxel_obs.dtype.names else None)
            else:
                block_names = None

            if block_names is None:
                return None

            bn_arr = np.asarray(block_names)
            if bn_arr.ndim < 3:
                return None

            X, Y, Z = bn_arr.shape
            cx, cy, cz = X // 2, Y // 2, Z // 2   # agent index in grid

            # ── Agent camera matrix from pitch / yaw ──────────────────────────
            loc = raw_obs.get("location_stats", {}) or {}
            def _flt(d, k, default=0.0):
                v = d.get(k, default)
                if v is None: return default
                try: return float(np.asarray(v).ravel()[0])
                except: return default

            yaw_deg   = _flt(loc, "yaw",   0.0)
            pitch_deg = _flt(loc, "pitch", 0.0)
            ya = math.radians(yaw_deg)
            pa = math.radians(pitch_deg)

            # Camera forward (+Z_cam = player look direction)
            # In world XYZ: fx=-sin(ya)*cos(pa), fy=-sin(pa), fz=cos(ya)*cos(pa)
            fw = np.array([
                -math.sin(ya) * math.cos(pa),
                -math.sin(pa),
                 math.cos(ya) * math.cos(pa),
            ], dtype=float)

            # Camera right (+X_cam = right in view)
            # right = normalize(cross(world_up, fw))
            world_up = np.array([0.0, 1.0, 0.0])
            rt = np.cross(world_up, fw)
            rt_norm = np.linalg.norm(rt)
            if rt_norm < 1e-6:
                rt = np.array([1.0, 0.0, 0.0])
            else:
                rt = rt / rt_norm

            # Camera up (+Y_cam = upward in view) = fw × right (left-hand rule)
            up = np.cross(fw, rt)

            # ── Project each target voxel into image coordinates ──────────────
            iy_list, ix_list = [], []
            for xi in range(X):
                for yi in range(Y):
                    for zi in range(Z):
                        name = str(bn_arr[xi, yi, zi]).lower().strip().rstrip("\x00")
                        if name not in targets:
                            continue

                        # World-relative offset (voxel centre; +0.5 = halfway into cell)
                        world_off = np.array([
                            float(xi - cx),
                            float(yi - cy),
                            float(zi - cz),
                        ], dtype=float)

                        # Camera-space components
                        fwd_dist = float(np.dot(fw, world_off))
                        if fwd_dist <= 0.1:  # behind or at agent
                            continue

                        x_cam = float(np.dot(rt, world_off))
                        y_cam = float(np.dot(up, world_off))

                        # Pinhole perspective (focal length = 0.5 normalised units)
                        # gives ~90° FoV matching Minecraft's 70° (close enough)
                        FOCAL = 1.0
                        u = (x_cam / (fwd_dist * FOCAL)) * 0.5 + 0.5   # [0,1]
                        v = (-y_cam / (fwd_dist * FOCAL)) * 0.5 + 0.5  # [0,1]; +y=down

                        img_x = int(np.clip(u * W, 0, W - 1))
                        img_y = int(np.clip(v * H, 0, H - 1))
                        iy_list.append(img_y)
                        ix_list.append(img_x)

            if not ix_list:
                # Target exists somewhere in grid but all behind player — use
                # full-frame so ROCKET-1 can turn around.
                return None

            # Paint blobs at projected target positions
            mask = np.zeros((H, W), dtype=np.uint8)
            blob = max(H // 8, 12)
            for iy, ix in zip(iy_list, ix_list):
                y0, y1 = max(0, iy - blob), min(H, iy + blob)
                x0, x1 = max(0, ix - blob), min(W, ix + blob)
                mask[y0:y1, x0:x1] = 1
            return mask

        except Exception as exc:
            logger.debug("_voxel_to_mask error (%s) — using fallback", exc)
            return None

    def _get_rgb_224(self, obs: dict) -> np.ndarray:
        """
        Extract a (224, 224, 3) uint8 C-contiguous numpy array from obs.

        The MinedojoObsWrapper downsizes to 64×64 internally, so we upscale.
        We apply np.ascontiguousarray to satisfy cv2's memory-layout requirement
        (transposed views from channel-first→channel-last conversion are
        non-contiguous and cause cv2.error "src is not a numpy array").
        """
        try:
            rgb = obs.get("rgb")
            if rgb is None:
                raise ValueError("no rgb key")
            # Force a proper C-contiguous uint8 numpy array
            rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
            # Handle channel-first (3, H, W) → (H, W, 3)
            if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[2] != 3:
                rgb = np.ascontiguousarray(rgb.transpose(1, 2, 0))
            if rgb.shape[:2] != (224, 224):
                import cv2
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            return rgb
        except Exception as exc:
            logger.debug("_get_rgb_224 fallback (%s)", exc)
            return np.zeros((224, 224, 3), dtype=np.uint8)
