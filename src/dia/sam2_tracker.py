"""
SAM-2 object tracker for ROCKET-1 visual grounding.

Wraps MineStudio's bundled realtime_sam (SAM-2 CameraPredictor) to generate
per-frame binary object masks that are fed as the 4th channel to ROCKET-1's
ViT backbone.

Pipeline
--------
1. At the first frame of each skill attempt:
   - Seed SAM-2 with a point click derived from the voxel-projection of the
     target block.  If no voxel hit is found in the image, fall back to a
     center-frame click.

2. Every subsequent frame:
   - Call predictor.track(pil_frame) → mask logits → binary (H, W) uint8 mask.
   - If the tracked mask collapses (< MIN_NONZERO pixels), re-seed using the
     current voxel projection (or center).

3. The mask is passed to ROCKET1GatherOption._make_mask(), which no longer
   needs to do projection itself when a SAM-2 tracker is available.

Usage
-----
    from dia.sam2_tracker import SAM2Tracker

    tracker = SAM2Tracker.build(device="cuda")   # lazy: None if SAM-2 unavailable
    if tracker is not None:
        tracker.reset()                           # call once per skill attempt
        mask = tracker.update(rgb_uint8, seed_xy=(cx, cy))  # (H, W) uint8
    else:
        mask = np.ones((H, W), dtype=np.uint8)   # fallback: full frame

Notes
-----
- SAM-2 C extension (_C) is NOT compiled in the dia-minecraft env.
  We build with apply_postprocessing=False to avoid the CUDA _C dependency.
- SAM-2's perpare_data() uses cv2.resize which is broken on numpy 2.x.
  We pass PIL Images to take the PIL code path instead.
- Thread safety: not thread-safe.  One tracker per option instance.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── SAM-2 installation paths ────────────────────────────────────────────────
# Installed by minestudio inside realtime_sam/
_MINESTUDIO_SAM2_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # src/dia/
    "../../",  # project root — won't work; use explicit path below
)

_SAM2_PACKAGE_DIR = None  # discovered lazily
_SAM2_CKPT_PATH   = None

def _discover_sam2() -> Tuple[Optional[str], Optional[str]]:
    """Return (sam2_package_dir, ckpt_path) or (None, None) if not available."""
    try:
        import minestudio  # noqa
        # minestudio.__file__ = .../site-packages/minestudio/__init__.py
        ms_root = os.path.dirname(minestudio.__file__)
        realtime = os.path.join(ms_root, "utils", "realtime_sam")
        sam2_pkg  = os.path.join(realtime, "sam2")
        ckpt_tiny  = os.path.join(realtime, "checkpoints", "sam2_hiera_tiny.pt")
        ckpt_small = os.path.join(realtime, "checkpoints", "sam2_hiera_small.pt")
        # Prefer tiny checkpoint to reduce GPU memory; fall back to small.
        ckpt = ckpt_tiny if os.path.isfile(ckpt_tiny) else ckpt_small
        if os.path.isdir(sam2_pkg) and os.path.isfile(ckpt):
            return sam2_pkg, ckpt
    except Exception:
        pass
    return None, None


# Minimum non-zero pixels to consider a tracked mask "alive"
_MIN_NONZERO = 50


class SAM2Tracker:
    """Per-skill SAM-2 tracking session.

    Attributes
    ----------
    predictor : SAM2CameraPredictor  (or None if build() fails)
    _initialized : bool — True after the first successful seed
    """

    def __init__(self, predictor) -> None:
        self._predictor = predictor
        self._initialized = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, device: str = "cuda") -> Optional["SAM2Tracker"]:
        """Load SAM-2 and return a tracker, or None if unavailable."""
        global _SAM2_PACKAGE_DIR, _SAM2_CKPT_PATH
        if _SAM2_PACKAGE_DIR is None:
            _SAM2_PACKAGE_DIR, _SAM2_CKPT_PATH = _discover_sam2()
        if _SAM2_PACKAGE_DIR is None:
            logger.warning(
                "SAM-2 not found.  Download sam2_hiera_tiny.pt (preferred) or "
                "sam2_hiera_small.pt to minestudio/utils/realtime_sam/checkpoints/"
            )
            return None

        try:
            # Hydra config is resolved relative to the sam2 package dir
            orig_dir = os.getcwd()
            os.chdir(_SAM2_PACKAGE_DIR)
            if _SAM2_PACKAGE_DIR not in sys.path:
                # Add parent (realtime_sam/) so `import sam2` resolves
                sys.path.insert(0, os.path.dirname(_SAM2_PACKAGE_DIR))
            from sam2.build_sam import build_sam2_camera_predictor  # type: ignore
            # Select config to match the checkpoint: tiny → t, small → s.
            _cfg_name = (
                "sam2_hiera_t.yaml"
                if os.path.basename(_SAM2_CKPT_PATH) == "sam2_hiera_tiny.pt"
                else "sam2_hiera_s.yaml"
            )
            predictor = build_sam2_camera_predictor(
                _cfg_name,
                _SAM2_CKPT_PATH,
                device=device,
                apply_postprocessing=False,  # skip _C CUDA extension
            )
            os.chdir(orig_dir)
            logger.info("SAM-2 camera predictor loaded on %s", device)
            return cls(predictor)
        except Exception as exc:
            try:
                os.chdir(orig_dir)
            except Exception:
                pass
            logger.warning("SAM-2 unavailable (%s)", exc)
            return None

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Call once per skill attempt to reset the tracking session."""
        self._initialized = False
        # predictor state is reset via load_first_frame on the next update()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        rgb_hwc: np.ndarray,
        seed_xy: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Get a binary (H, W) uint8 object mask for this frame.

        Parameters
        ----------
        rgb_hwc : np.ndarray, shape (H, W, 3), dtype uint8
            Current RGB game frame.
        seed_xy : (x, y) pixel coordinate for object seeding (image coords).
            None means "target not currently visible" — returns zero mask so
            ROCKET-1 enters exploration/search mode rather than fixating on
            a false center-point guess.

        Returns
        -------
        np.ndarray, shape (H, W), dtype uint8 — 1 = object pixels.
        Zero mask = target not yet visible (search mode).
        """
        H, W = rgb_hwc.shape[:2]
        try:
            from PIL import Image as _PILImage

            pil_frame = _PILImage.fromarray(rgb_hwc)

            if not self._initialized:
                if seed_xy is None:
                    # Target not visible in this frame — return zero mask so
                    # ROCKET-1 knows to look around rather than mine nothing.
                    return np.zeros((H, W), dtype=np.uint8)
                # ── Seed SAM-2 with the first frame ──────────────────────
                self._predictor.load_first_frame(pil_frame)
                cx, cy = seed_xy
                points = np.array([[cx, cy]], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)
                _, _out_ids, out_mask_logits = self._predictor.add_new_prompt(
                    frame_idx=0, obj_id=1, points=points, labels=labels
                )
                self._initialized = True
                # Use the actual SAM-2 mask from add_new_prompt (frame 0 output)
                if out_mask_logits is not None and len(out_mask_logits) > 0:
                    raw_mask = (
                        out_mask_logits[0] > 0.0
                    ).squeeze(0).cpu().numpy().astype(np.uint8)
                    if raw_mask.shape != (H, W):
                        raw_mask = self._resize_mask(raw_mask, H, W)
                    if int(raw_mask.sum()) >= _MIN_NONZERO:
                        return raw_mask
                # SAM-2 returned empty mask (seed hit nothing useful) —
                # fall back to blob so ROCKET-1 at least has some spatial signal
                return self._blob_mask(H, W, cy, cx)

            # ── Track subsequent frames ───────────────────────────────────
            out_ids, out_logits = self._predictor.track(pil_frame)
            if out_logits is None or len(out_logits) == 0:
                self._initialized = False
                return np.zeros((H, W), dtype=np.uint8)

            raw_mask = (out_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
            # raw_mask is at SAM-2 internal resolution (1024×1024); resize to (H, W)
            if raw_mask.shape != (H, W):
                raw_mask = self._resize_mask(raw_mask, H, W)

            if int(raw_mask.sum()) < _MIN_NONZERO:
                # Tracking collapsed — reset, return zero mask (search mode)
                self._initialized = False
                logger.debug("SAM-2 mask collapsed; entering search mode")
                return np.zeros((H, W), dtype=np.uint8)

            return raw_mask

        except Exception as exc:
            logger.debug("SAM2Tracker.update error (%s) — full frame fallback", exc)
            return self._full_frame(H, W)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _blob_mask(H: int, W: int, cy: int, cx: int, radius: int = 0) -> np.ndarray:
        mask = np.zeros((H, W), dtype=np.uint8)
        r = radius if radius > 0 else max(H // 8, 16)
        y0, y1 = max(0, cy - r), min(H, cy + r)
        x0, x1 = max(0, cx - r), min(W, cx + r)
        mask[y0:y1, x0:x1] = 1
        return mask

    @staticmethod
    def _full_frame(H: int, W: int) -> np.ndarray:
        return np.ones((H, W), dtype=np.uint8)

    @staticmethod
    def _resize_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
        """Nearest-neighbour resize using numpy (avoids broken cv2)."""
        from PIL import Image as _PIL
        return (np.array(
            _PIL.fromarray(mask * 255).resize((W, H), _PIL.NEAREST)
        ) > 127).astype(np.uint8)
