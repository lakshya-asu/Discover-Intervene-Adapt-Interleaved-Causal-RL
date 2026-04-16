"""
Florence-2 visual grounder for SAM-2 seeding in ROCKET-1 gather skills.

Replaces hardcoded _SEED_FRACTIONS heuristic with open-vocabulary detection:
  text prompt -> Florence-2 -> bounding box -> center -> SAM-2 seed point

Only called when SAM-2 needs a new seed (skill start or tracking collapse),
NOT every frame. Typical latency: ~15-20ms on RTX 3090.

Compatibility note
------------------
The cached microsoft/Florence-2-base model code was written for transformers
< 4.40 where PretrainedConfig carried a forced_bos_token_id attribute.
In transformers >= 4.40 that field moved to GenerationConfig, so accessing
it directly on a PretrainedConfig subclass raises AttributeError.  build()
applies a minimal monkey-patch to PretrainedConfig.__getattribute__ during
the load call only, restoring the safe default (None) when the attribute is
absent.  The patch is removed immediately after loading.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Text prompts per skill — what to look for in the image
_SKILL_PROMPTS = {
    "wood":    "oak log wood tree trunk",
    "stone":   "stone cobblestone rock",
    "coal":    "coal ore dark mineral",
    "ironore": "iron ore mineral",
    "diamond": "diamond ore cyan mineral",
}


def _patch_pretrained_config_getattribute():
    """Temporary shim for transformers >= 4.40 / Florence-2 cached code mismatch.

    Returns the original __getattribute__ so the caller can restore it.
    """
    try:
        from transformers import configuration_utils as _cu
        _orig = _cu.PretrainedConfig.__getattribute__

        def _patched(self, key):
            if key == "forced_bos_token_id":
                try:
                    return _orig(self, key)
                except AttributeError:
                    return None
            return _orig(self, key)

        _cu.PretrainedConfig.__getattribute__ = _patched
        return _cu, _orig
    except Exception:
        return None, None


def _restore_pretrained_config_getattribute(cu, orig):
    """Restore __getattribute__ after loading."""
    if cu is not None and orig is not None:
        try:
            cu.PretrainedConfig.__getattribute__ = orig
        except Exception:
            pass


def _patch_transformers_5x_compat() -> None:
    """Apply all shims needed for Florence-2 cached code under transformers 5.x.

    Florence-2 was written for transformers ~4.38; several attributes were
    removed or renamed in 5.x.  Each patch is the minimal fix:

    1. additional_special_tokens — property removed from PreTrainedTokenizerBase;
       now backed by _extra_special_tokens (defaults to []).
    2. _supports_sdpa — class attribute removed from PreTrainedModel defaults;
       Florence-2 model class references it during from_pretrained; default False.
    """
    try:
        from transformers import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
            PreTrainedTokenizerBase.additional_special_tokens = property(
                lambda self: getattr(self, "_extra_special_tokens", [])
            )
    except Exception:
        pass

    try:
        from transformers import PreTrainedModel
        if not hasattr(PreTrainedModel, "_supports_sdpa"):
            PreTrainedModel._supports_sdpa = False
    except Exception:
        pass


class FlorenceGrounder:
    """Open-vocabulary visual grounder using Florence-2-base."""

    MODEL_ID = "microsoft/Florence-2-base"

    def __init__(self, model, processor, device):
        self._model = model
        self._processor = processor
        self._device = device

    @classmethod
    def build(cls, device: str = "cuda") -> Optional["FlorenceGrounder"]:
        """Load Florence-2-base. Returns None if unavailable."""
        # Shim 1: forced_bos_token_id removed in transformers >= 4.40
        cu, orig = _patch_pretrained_config_getattribute()
        # Shim 2+3: additional_special_tokens + _supports_sdpa removed in transformers >= 5.x
        _patch_transformers_5x_compat()
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM

            processor = AutoProcessor.from_pretrained(
                cls.MODEL_ID, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                cls.MODEL_ID,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="eager",  # skip SDPA check (Florence-2 @property bug)
            ).to(device).eval()
            logger.info("Florence-2-base loaded on %s", device)
            return cls(model, processor, device)
        except Exception as exc:
            logger.warning("FlorenceGrounder.build failed: %s", exc)
            return None
        finally:
            _restore_pretrained_config_getattribute(cu, orig)

    def point(
        self,
        rgb_hwc: np.ndarray,
        skill_name: str,
    ) -> Optional[Tuple[int, int]]:
        """
        Detect the target object and return its center pixel (x, y), or None.

        Uses Florence-2's OPEN_VOCABULARY_DETECTION task with a skill-specific
        text prompt. Returns the center of the highest-confidence bounding box,
        or None if no detection found.
        """
        try:
            import torch
            from PIL import Image as _PIL

            prompt = _SKILL_PROMPTS.get(skill_name, skill_name)
            H, W = rgb_hwc.shape[:2]
            pil_img = _PIL.fromarray(rgb_hwc)

            task = "<OPEN_VOCABULARY_DETECTION>"
            inputs = self._processor(
                text=task + prompt,
                images=pil_img,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                outputs = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"].to(torch.float16),
                    max_new_tokens=256,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                )

            result_text = self._processor.batch_decode(
                outputs, skip_special_tokens=False
            )[0]
            parsed = self._processor.post_process_generation(
                result_text,
                task=task,
                image_size=(W, H),
            )

            detections = parsed.get(task, {})
            bboxes = detections.get("bboxes", [])
            if not bboxes:
                logger.debug("[%s] Florence-2: no detection", skill_name)
                return None

            # Use center of first (highest-confidence) bounding box
            x0, y0, x1, y1 = bboxes[0]
            cx = int((x0 + x1) / 2)
            cy = int((y0 + y1) / 2)
            logger.debug(
                "[%s] Florence-2 detected at (%d,%d) bbox=%s",
                skill_name, cx, cy, bboxes[0],
            )
            return (cx, cy)

        except Exception as exc:
            logger.debug("FlorenceGrounder.point error: %s", exc)
            return None
