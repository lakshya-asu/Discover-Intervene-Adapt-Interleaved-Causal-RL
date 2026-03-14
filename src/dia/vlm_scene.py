"""
vlm_scene.py
============
VLM-based scene analyzer for Montezuma's Revenge using Claude claude-opus-4-6.

Uses the Anthropic ``tool_use`` mechanism to produce **guaranteed structured
output** that maps directly onto the 8 DIA/EVGS variables.  Each variable is
returned as a soft probability in [0, 1], giving richer training signal than
the binary RAM-based readings while remaining semantically aligned.

Usage
-----
    from dia.vlm_scene import VLMSceneAnalyzer, VLMSceneConfig

    analyzer = VLMSceneAnalyzer(VLMSceneConfig(call_every=5, verbose=True))
    result = analyzer.analyze(frame_rgb)      # np.ndarray (210, 160, 3)
    print(result.scene_description)
    print(result.to_array())                  # shape (8,), values in [0, 1]

Environment
-----------
Set ``CLAUDE_API_KEY`` **or** ``ANTHROPIC_API_KEY`` before calling ``analyze()``.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ── Variable names must match MontezumaRichWrapper.VAR_NAMES order ──────────
MONTEZUMA_VAR_NAMES: List[str] = [
    "has_key",
    "door_open",
    "at_key_zone",
    "at_door_zone",
    "on_upper_level",
    "near_rope",
    "skull_near",
    "score_gained",
]

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a precise game-state extractor for Montezuma's Revenge (Atari 2600).

Room 1 layout reference
-----------------------
• Player   − small orange humanoid figure
• Key      − bright yellow/gold key shape, upper-right platform
• Door     − bottom-left area of the screen
• Skull    − white bouncing skull on the middle platform (moves left/right)
• Rope     − vertical hanging rope, roughly centre of screen

Coordinate convention: pixel Y increases downward.
  top platforms   ≈ Y < 165
  middle platform ≈ Y 165-215   (rope lives here)
  bottom floor    ≈ Y 215-250

Confidence scale: 0.0 = definitely not present/true, 1.0 = definitely true.
For uncertain cases (partially occluded, ambiguous position) use 0.3–0.7.
"""

# ── Anthropic tool schema ─────────────────────────────────────────────────────
_TOOL_SCHEMA: Dict = {
    "name": "report_scene_state",
    "description": (
        "Report the quantified game state from this Montezuma's Revenge frame. "
        "All numeric fields must be in [0.0, 1.0]."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "has_key", "door_open", "at_key_zone", "at_door_zone",
            "on_upper_level", "near_rope", "skull_near", "score_gained",
            "scene_description", "player_action",
        ],
        "properties": {
            "has_key": {
                "type": "number",
                "description": (
                    "0–1 confidence that the player is currently holding/has collected "
                    "the key (key glyph absent from environment, or score recently jumped "
                    "by ~100 pts)."
                ),
            },
            "door_open": {
                "type": "number",
                "description": (
                    "0–1 confidence that the door obstacle has been cleared/opened (visible "
                    "change in the door area or score jumped by ~300 pts)."
                ),
            },
            "at_key_zone": {
                "type": "number",
                "description": (
                    "0–1 confidence that the player is on or immediately adjacent to the "
                    "upper-right key platform where the key resides."
                ),
            },
            "at_door_zone": {
                "type": "number",
                "description": (
                    "0–1 confidence that the player is in the bottom-left door area."
                ),
            },
            "on_upper_level": {
                "type": "number",
                "description": (
                    "0–1 confidence that the player is on any platform above the bottom "
                    "floor (middle or top platforms, roughly pixel-Y < 210)."
                ),
            },
            "near_rope": {
                "type": "number",
                "description": (
                    "0–1 confidence that the player is on or directly adjacent to the "
                    "central vertical rope (roughly screen centre-left)."
                ),
            },
            "skull_near": {
                "type": "number",
                "description": (
                    "0–1 confidence that a skull enemy is within approximately 30 pixels "
                    "of the player character."
                ),
            },
            "score_gained": {
                "type": "number",
                "description": (
                    "0–1 confidence that the score display has increased since the "
                    "last frame (any non-zero reward this step)."
                ),
            },
            "scene_description": {
                "type": "string",
                "description": (
                    "One concise sentence (≤20 words) describing the most salient "
                    "features of the scene."
                ),
            },
            "player_action": {
                "type": "string",
                "enum": ["standing", "climbing", "jumping", "running", "falling", "unknown"],
                "description": "Best guess at the player's current movement state.",
            },
        },
    },
}


# ── Config / Result dataclasses ───────────────────────────────────────────────

@dataclass
class VLMSceneConfig:
    """Configuration for the VLM scene analyzer."""

    model: str = "claude-opus-4-6"
    """Anthropic model to call."""

    max_tokens: int = 512
    """Maximum tokens to generate per API call."""

    call_every: int = 200
    """Call the VLM every *N* primitive steps.
    One option typically runs ~160 primitive steps, so the default of 200
    means roughly one VLM call per option execution.  All intermediate
    extract() calls reuse the cached result at zero API cost."""

    frame_scale: int = 2
    """Integer upscale factor applied to the Atari frame before sending
    to the VLM.  ``2`` converts 210×160 → 420×320 which makes individual
    sprites easier for the model to identify."""

    api_key_envs: List[str] = field(
        default_factory=lambda: ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"]
    )
    """Environment variable names searched in order for the API key."""

    verbose: bool = False
    """Print a one-line summary after each successful API call.
    Errors are printed only on the *first* occurrence to avoid spam."""


@dataclass
class VLMSceneResult:
    """Structured output of one VLM scene analysis call."""

    # Core variables — float probabilities in [0, 1]
    has_key:        float = 0.0
    door_open:      float = 0.0
    at_key_zone:    float = 0.0
    at_door_zone:   float = 0.0
    on_upper_level: float = 0.0
    near_rope:      float = 0.0
    skull_near:     float = 0.0
    score_gained:   float = 0.0

    # Qualitative context
    scene_description: str = ""
    player_action:     str = "unknown"

    # Metadata
    error:      Optional[str] = None
    latency_ms: float         = 0.0

    # ------------------------------------------------------------------ #

    def to_array(self, var_names: Optional[List[str]] = None) -> np.ndarray:
        """Return VLM probabilities as ndarray in EVGS variable order."""
        names = var_names or MONTEZUMA_VAR_NAMES
        return np.array(
            [float(getattr(self, n, 0.0)) for n in names],
            dtype=float,
        )

    def as_info_addendum(self) -> Dict[str, object]:
        """Dict of ``vlm_*`` keys suitable for merging into an info dict."""
        out: Dict[str, object] = {
            f"vlm_{n}": float(getattr(self, n, 0.0))
            for n in MONTEZUMA_VAR_NAMES
        }
        out["vlm_scene"]  = self.scene_description
        out["vlm_action"] = self.player_action
        return out


# ── Main analyzer class ───────────────────────────────────────────────────────

class VLMSceneAnalyzer:
    """
    Calls Claude claude-opus-4-6 (vision) to estimate the 8 EVGS variables
    from a raw Atari game frame.

    Responses are cached for ``cfg.call_every`` steps to keep API costs
    manageable.  On any error (network, auth, parse) the analyzer returns a
    ``VLMSceneResult`` with ``error`` set, so callers can fall back gracefully.
    """

    def __init__(self, cfg: Optional[VLMSceneConfig] = None) -> None:
        self.cfg = cfg or VLMSceneConfig()
        self._client      = None
        self._step        = 0
        self._cache: Optional[VLMSceneResult] = None

        # Stats
        self._total_calls:   int   = 0   # successful API calls
        self._total_latency: float = 0.0
        self._error_count:   int   = 0
        self._attempt_count: int   = 0   # total _call_vlm() attempts

        # Error suppression — only print each unique error message once
        self._seen_errors: set = set()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze(self, frame: Optional[np.ndarray]) -> VLMSceneResult:
        """
        Return a ``VLMSceneResult`` for *frame*.

        First call and every ``call_every`` calls thereafter invoke the API.
        All other calls return the cached result.
        """
        self._step += 1
        if self._cache is not None and (self._step % self.cfg.call_every) != 1:
            return self._cache

        if frame is None:
            return VLMSceneResult(error="no frame provided")

        result = self._call_vlm(frame)
        self._cache = result
        return result

    def stats(self) -> Dict[str, float]:
        """Return a summary of API usage metrics."""
        attempts = max(1, self._attempt_count)
        return {
            "total_vlm_calls":  float(self._total_calls),
            "total_attempts":   float(self._attempt_count),
            "avg_latency_ms":   self._total_latency / max(1, self._total_calls),
            "error_rate":       self._error_count / attempts,
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pip install anthropic  to use VLM enrichment"
            ) from exc

        api_key: Optional[str] = None
        for env_var in self.cfg.api_key_envs:
            api_key = os.environ.get(env_var)
            if api_key:
                break

        if not api_key:
            raise RuntimeError(
                "No Anthropic API key found. "
                f"Set one of: {self.cfg.api_key_envs}"
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Convert numpy RGB uint8 array → base64 PNG string (upscaled)."""
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install Pillow  to use VLM enrichment") from exc

        img = Image.fromarray(frame.astype(np.uint8))
        if self.cfg.frame_scale > 1:
            w, h = img.size
            img = img.resize(
                (w * self.cfg.frame_scale, h * self.cfg.frame_scale),
                Image.NEAREST,
            )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_vlm(self, frame: np.ndarray) -> VLMSceneResult:
        """Make one Anthropic API call and parse the tool_use response."""
        t0 = time.time()
        try:
            client = self._get_client()
            b64    = self._encode_frame(frame)

            response = client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=_SYSTEM_PROMPT,
                tools=[_TOOL_SCHEMA],
                tool_choice={"type": "any"},   # force tool use; never plain text
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type":       "base64",
                                    "media_type": "image/png",
                                    "data":       b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this Montezuma's Revenge frame and "
                                    "call report_scene_state with your findings."
                                ),
                            },
                        ],
                    }
                ],
            )

            # ── Guard: proxy environments return raw SSE instead of Message ─
            if isinstance(response, str):
                raise RuntimeError(
                    "Anthropic API returned a raw SSE string instead of a "
                    "Message object — likely ANTHROPIC_API_KEY is a proxy key. "
                    "Set CLAUDE_API_KEY to your personal Anthropic API key."
                )

            # ── Parse tool_use block ──────────────────────────────────────
            tool_input = None
            for block in response.content:
                if (
                    getattr(block, "type", None) == "tool_use"
                    and block.name == "report_scene_state"
                ):
                    tool_input = block.input
                    break

            if tool_input is None:
                return VLMSceneResult(
                    error="no tool_use block in response",
                    latency_ms=(time.time() - t0) * 1000,
                )

            # ── Build result ──────────────────────────────────────────────
            float_fields = [
                "has_key", "door_open", "at_key_zone", "at_door_zone",
                "on_upper_level", "near_rope", "skull_near", "score_gained",
            ]
            latency = (time.time() - t0) * 1000
            self._total_calls   += 1
            self._attempt_count += 1
            self._total_latency += latency

            result = VLMSceneResult(
                **{
                    k: float(np.clip(float(tool_input.get(k, 0.0)), 0.0, 1.0))
                    for k in float_fields
                },
                scene_description=str(tool_input.get("scene_description", "")),
                player_action=str(tool_input.get("player_action", "unknown")),
                latency_ms=latency,
            )

            if self.cfg.verbose:
                print(
                    f"  [VLM] {result.scene_description}  "
                    f"action={result.player_action}  ({latency:.0f} ms)"
                )

            return result

        except Exception as exc:
            self._error_count   += 1
            self._attempt_count += 1
            latency = (time.time() - t0) * 1000
            err_msg = str(exc)
            if self.cfg.verbose and err_msg not in self._seen_errors:
                print(f"  [VLM] error ({latency:.0f} ms): {err_msg}  (will not repeat)")
                self._seen_errors.add(err_msg)
            return VLMSceneResult(error=err_msg, latency_ms=latency)
