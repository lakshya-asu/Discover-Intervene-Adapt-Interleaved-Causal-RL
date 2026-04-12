"""
clip_skill.py
=============
CLIP-based goal encoding for MineDojo skill training and inference.

Provides two things:

1. **SKILL_TEXTS** — natural-language descriptions of each DIA variable's
   acquisition task.  These are used as frozen CLIP text embeddings during
   PPO training (no API cost) and can be overridden at inference by VLM-
   generated descriptions.

2. **CLIPGoalEncoder** — thin wrapper around OpenAI CLIP (ViT-B/32) that:
   - Encodes text goals → normalised float32 [512] vectors.
   - Encodes RGB frames → normalised float32 [512] vectors.
   - Computes cosine-similarity reward ∈ [0, 1] between frame and goal.

The encoder is loaded lazily and cached as a module-level singleton so that
multiple wrappers share a single GPU/CPU allocation.

Dependencies
------------
  pip install openai-clip          # provides `clip` module
  pip install Pillow               # for image pre-processing

Usage
-----
    from dia.clip_skill import CLIPGoalEncoder, SKILL_TEXTS

    enc = CLIPGoalEncoder.get()                    # singleton
    embed = enc.skill_embedding("wood")            # [512] float32
    reward = enc.clip_reward(frame_rgb, embed)     # float in [0, 1]
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional


# ── Per-skill text descriptions ───────────────────────────────────────────────
# Written to be visually grounded: describe what the agent *sees* when it is
# successfully executing the skill, not an abstract goal.  This helps CLIP
# match the description to the correct visual content.

SKILL_TEXTS: Dict[str, str] = {
    "wood":          "chopping a tree trunk with bare hands to collect wood logs",
    "stone":         "mining grey stone blocks with a pickaxe underground",
    "coal":          "mining black coal ore patches visible in stone",
    "ironore":       "mining reddish-brown iron ore embedded in stone",
    "furnace":       "placing a stone furnace block and smelting items",
    "stonepickaxe":  "crafting a stone pickaxe at a crafting table",
    "iron":          "smelting iron ingots in a lit furnace",
    "ironpickaxe":   "crafting an iron pickaxe at a crafting table",
    "diamond":       "mining blue diamond ore with an iron pickaxe deep underground",
}

GOAL_DIM = 512   # ViT-B/32 text embedding dimension


# ── Singleton CLIP encoder ────────────────────────────────────────────────────

class CLIPGoalEncoder:
    """
    Lazily-loaded, module-level singleton wrapping OpenAI CLIP ViT-B/32.

    Typical use
    -----------
        enc = CLIPGoalEncoder.get()            # shared instance
        text_embed = enc.skill_embedding("wood")
        clip_r     = enc.clip_reward(frame, text_embed)
    """

    _instance: Optional["CLIPGoalEncoder"] = None

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu") -> None:
        try:
            import clip            # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pip install openai-clip  to use CLIP goal conditioning"
            ) from exc

        import torch
        self.device = device
        self._model, self._preprocess = clip.load(model_name, device=device)
        self._model.eval()
        self._clip = clip
        self._torch = torch
        self._text_cache: Dict[str, np.ndarray] = {}

    # ── Singleton factory ─────────────────────────────────────────────────

    @classmethod
    def get(cls, device: str = "cpu") -> "CLIPGoalEncoder":
        """Return the shared instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode_text(self, text: str) -> np.ndarray:
        """Return a normalised [512] float32 embedding for *text*."""
        if text in self._text_cache:
            return self._text_cache[text]

        tokens = self._clip.tokenize([text]).to(self.device)
        with self._torch.no_grad():
            feat = self._model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        arr = feat.cpu().numpy()[0].astype(np.float32)
        self._text_cache[text] = arr
        return arr

    def encode_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a normalised [512] float32 embedding for a H×W×3 uint8 frame.
        """
        from PIL import Image  # type: ignore

        img = Image.fromarray(frame.astype(np.uint8))
        tensor = self._preprocess(img).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            feat = self._model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].astype(np.float32)

    # ── Reward ────────────────────────────────────────────────────────────

    def clip_reward(self, frame: np.ndarray, text_embed: np.ndarray) -> float:
        """
        Cosine similarity between *frame* and *text_embed*, mapped to [0, 1].

        Parameters
        ----------
        frame       : H×W×3 uint8 numpy array
        text_embed  : [512] float32, normalised (from encode_text / skill_embedding)

        Returns
        -------
        float in [0, 1]  (0.5 = orthogonal, 1.0 = perfectly aligned)
        """
        img_embed = self.encode_image(frame)
        cosine = float(np.dot(img_embed, text_embed))  # both unit vectors
        return (cosine + 1.0) / 2.0                    # shift [-1,1] → [0,1]

    def skill_embedding(self, var_name: str) -> np.ndarray:
        """Return CLIP text embedding for the named DIA variable's skill."""
        text = SKILL_TEXTS.get(var_name, f"gathering {var_name} in Minecraft")
        return self.encode_text(text)

    # ── Batch helpers for VLM-supplied text ──────────────────────────────

    def encode_vlm_instruction(self, instruction: str) -> np.ndarray:
        """
        Encode a free-form VLM instruction string (e.g. from MineRLVLMAnalyzer)
        the same way as skill texts.  The policy was trained on SKILL_TEXTS
        embeddings, but because CLIP shares a universal embedding space the
        VLM text should land nearby when it describes the same visual action.
        """
        return self.encode_text(instruction)
