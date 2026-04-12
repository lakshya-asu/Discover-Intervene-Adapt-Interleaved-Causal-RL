# src/dia/vlm_minerl.py
"""
Claude Opus 4.6 visual consultant for MineRL 3D Minecraft.

Parallel to vlm_scene.py (Montezuma's Revenge), but tuned for the 9-variable
DIA Minecraft tech tree: wood→stone→coal→ironore→furnace→stonepickaxe→iron→ironpickaxe→diamond.

Called only at skill START and skill FAILURE — not every step.
One API call ≈ 1–3 seconds; rate-limiting to once per skill attempt keeps cost low.

Key outputs
-----------
visual_vars     : dict[str, float] — visual presence [0,1] per DIA variable
                  (what is *visible* in frame, not what is in inventory)
directive_actions: list[int] — 3-8 action indices from options_minerl._ACTION_DEFS
                  (recommended first moves before falling back to macro)
reason          : str — one sentence explanation

Action index legend (from options_minerl._ACTION_DEFS):
  0=noop  1=forward  2=back  3=left  4=right  5=jump
  6=attack  7=use  8=fwd+attack  9=fwd+jump  10=sprint_fwd
  11=cam_up  12=cam_down  13=cam_left  14=cam_right
  15=sprint+attack  16=cam_up_fine  17=cam_down_fine
  18=cam_left_fine  19=cam_right_fine  20=hotbar1  21=hotbar2
  22=fwd+use  23=jump+attack  24=fwd+jump+attack

Usage
-----
    from dia.vlm_minerl import MineRLVLMAnalyzer, MineRLVLMConfig

    analyzer = MineRLVLMAnalyzer(MineRLVLMConfig(verbose=True))
    result = analyzer.analyze(frame_rgb, var_name="wood", context="start")
    print(result.reason)
    print(result.directive_actions)   # e.g. [11, 11, 8, 8, 8, 6, 6, 6, 14, 14]

Set ANTHROPIC_API_KEY or CLAUDE_API_KEY before use.
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ── DIA variable names (9-variable Minecraft tech tree) ───────────────────────
MINERL_VAR_NAMES: List[str] = [
    "wood", "stone", "coal", "ironore", "furnace",
    "stonepickaxe", "iron", "ironpickaxe", "diamond",
]

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert Minecraft survival agent advisor.
You analyze a first-person 3D Minecraft screenshot and give precise, actionable advice.

The agent has a discrete 25-action set. Key action indices:
  1=walk_forward   2=walk_back     6=attack(mine/chop)  7=use(interact)
  8=forward+attack 9=forward+jump  10=sprint_forward    11=look_UP
  12=look_DOWN     13=look_LEFT    14=look_RIGHT        15=sprint+attack
  16=look_up_fine  17=look_down_fine                    22=forward+use
  24=forward+jump+attack

Resource gathering rules:
- WOOD: tree trunks need attack while looking at them (slightly above horizontal)
- STONE: cobblestone; look down slightly to hit ground or cliff face, then attack
- COAL: dark ore patches in stone walls; approach and attack stone face
- IRON_ORE: reddish patches in stone underground; same as coal
- FURNACE/PICKAXE crafting: requires inventory items + crafting table or crafting menu (use key)
- DIAMOND: deep underground (far below surface), needs iron pickaxe to mine

Always output a directive_actions list of 5-10 action indices as your first recommended moves.
These will be executed before the agent's default macro, so make them count.
"""

# ── Tool schema ────────────────────────────────────────────────────────────────
_TOOL_SCHEMA: Dict = {
    "name": "report_minecraft_scene",
    "description": (
        "Report visual scene analysis and action directive for a 3D Minecraft frame. "
        "All numeric fields in [0.0, 1.0]. directive_actions must be a list of 5-10 integers."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "wood_visible", "stone_visible", "coal_visible", "ironore_visible",
            "furnace_visible", "crafting_table_visible",
            "open_space_ahead", "underground",
            "directive_actions", "reason",
        ],
        "properties": {
            "wood_visible": {
                "type": "number",
                "description": "0-1 confidence that tree trunks or leaves are visible and reachable",
            },
            "stone_visible": {
                "type": "number",
                "description": "0-1 confidence that mineable stone/cobblestone surface is visible",
            },
            "coal_visible": {
                "type": "number",
                "description": "0-1 confidence that coal ore (dark patches in stone) is visible",
            },
            "ironore_visible": {
                "type": "number",
                "description": "0-1 confidence that iron ore (reddish patches in stone) is visible",
            },
            "furnace_visible": {
                "type": "number",
                "description": "0-1 confidence that a placed furnace block is visible nearby",
            },
            "crafting_table_visible": {
                "type": "number",
                "description": "0-1 confidence that a crafting table block is visible nearby",
            },
            "open_space_ahead": {
                "type": "number",
                "description": "0-1 confidence that the agent can walk forward without immediate obstacle",
            },
            "underground": {
                "type": "number",
                "description": "0-1 confidence that the agent is underground (cave/mine) vs on surface",
            },
            "directive_actions": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 24},
                "minItems": 5,
                "maxItems": 12,
                "description": (
                    "Ordered list of 5-12 action indices to execute first, before the default macro. "
                    "Prioritize getting the agent facing and approaching the target resource. "
                    "Example for wood: [11, 11, 8, 8, 8, 6, 6, 6, 14, 14] "
                    "(look up, walk+attack, attack, turn to search)"
                ),
            },
            "reason": {
                "type": "string",
                "description": "One sentence (≤25 words) explaining what you see and why you chose these actions.",
            },
        },
    },
}

# ── Per-skill context strings (inserted into the user prompt) ─────────────────
_SKILL_CONTEXT: Dict[str, str] = {
    "wood": (
        "Goal: chop a tree to get wood logs. "
        "Look for tree trunks (brown columns) and leaves (green clusters). "
        "Aim at the trunk, not the leaves. Trees are taller than 1 block."
    ),
    "stone": (
        "Goal: mine cobblestone. Look for stone/rock surfaces. "
        "Stone is grey. Looking down slightly hits ground-level stone. "
        "Looking straight ahead hits cliff faces."
    ),
    "coal": (
        "Goal: find and mine coal ore. Coal ore looks like grey stone with dark/black speckles. "
        "It appears in stone walls and cliffs. Approach the stone face and attack."
    ),
    "ironore": (
        "Goal: find and mine iron ore. Iron ore is grey stone with orange-brown speckles. "
        "It appears in cave walls and underground. Need to go deeper."
    ),
    "furnace": (
        "Goal: craft a furnace (requires 8 cobblestone in inventory). "
        "Open crafting menu with 'use' key or find a crafting table. "
        "If a furnace block is placed nearby, interact with it."
    ),
    "stonepickaxe": (
        "Goal: craft a stone pickaxe (requires 3 cobblestone + 2 sticks in inventory). "
        "Open crafting menu with 'use' key or approach a crafting table and interact."
    ),
    "iron": (
        "Goal: smelt iron ore into iron ingots. "
        "Need a furnace, iron ore, and coal in inventory. "
        "Find or place a furnace and interact with it (use key)."
    ),
    "ironpickaxe": (
        "Goal: craft an iron pickaxe (requires 3 iron ingots + 2 sticks in inventory). "
        "Open crafting menu or find a crafting table."
    ),
    "diamond": (
        "Goal: mine diamond ore (requires iron pickaxe). "
        "Diamonds are deep underground (y=5-12). Dig down aggressively. "
        "Diamond ore is blue-speckled grey stone."
    ),
}

# ── Config / Result dataclasses ───────────────────────────────────────────────

@dataclass
class MineRLVLMConfig:
    model: str = "claude-opus-4-6"
    max_tokens: int = 512
    frame_scale: int = 1        # 360×640 is already large; no upscale needed
    verbose: bool = False
    api_key_envs: List[str] = field(
        default_factory=lambda: ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]
    )


@dataclass
class MineRLVLMResult:
    # Visual presence estimates (inventory is separate — this is what you *see*)
    wood_visible:            float = 0.0
    stone_visible:           float = 0.0
    coal_visible:            float = 0.0
    ironore_visible:         float = 0.0
    furnace_visible:         float = 0.0
    crafting_table_visible:  float = 0.0
    open_space_ahead:        float = 0.5
    underground:             float = 0.0

    # Action directive and explanation
    directive_actions: List[int] = field(default_factory=list)
    reason: str = ""

    # Metadata
    error:      Optional[str] = None
    latency_ms: float         = 0.0

    # ── Visual vars as array ──────────────────────────────────────────────────

    def visual_var_dict(self) -> Dict[str, float]:
        """
        Map visual presence estimates to DIA variable names.
        These are visual estimates only — not inventory values.
        """
        return {
            "wood":          self.wood_visible,
            "stone":         self.stone_visible,
            "coal":          self.coal_visible,
            "ironore":       self.ironore_visible,
            "furnace":       self.furnace_visible,
            # crafting items have no direct visual presence — 0 unless crafting table visible
            "stonepickaxe":  0.0,
            "iron":          0.0,
            "ironpickaxe":   0.0,
            "diamond":       0.0,
        }

    def visual_var_array(self) -> np.ndarray:
        """Return visual presence as [9] float array in DIA VAR_NAMES order."""
        d = self.visual_var_dict()
        return np.array([d[n] for n in MINERL_VAR_NAMES], dtype=float)


# ── Main analyzer ─────────────────────────────────────────────────────────────

class MineRLVLMAnalyzer:
    """
    Calls Claude Opus 4.6 vision to analyze a MineRL POV frame.

    Designed to be called at skill START and FAILURE only (not every step).
    Rate-limited internally: two consecutive calls within `min_interval_s`
    seconds return the cached result.

    Parameters
    ----------
    cfg : MineRLVLMConfig
    min_interval_s : float
        Minimum seconds between API calls per skill name (prevents spam
        if the caller's loop calls analyze() very frequently).
    """

    def __init__(self, cfg: Optional[MineRLVLMConfig] = None,
                 min_interval_s: float = 2.0):
        self.cfg = cfg or MineRLVLMConfig()
        self._client = None
        self._min_interval = min_interval_s

        # Per-skill cache: var_name → (timestamp, MineRLVLMResult)
        self._cache: Dict[str, tuple] = {}

        # Stats
        self._total_calls   = 0
        self._total_latency = 0.0
        self._error_count   = 0
        self._seen_errors: set = set()

    # ── Public API ───────────────────────────────────────────────────────────

    def analyze(
        self,
        frame: Optional[np.ndarray],
        var_name: str = "wood",
        context: str = "start",     # "start" | "failed"
    ) -> MineRLVLMResult:
        """
        Analyze a MineRL POV frame for the given skill.

        Parameters
        ----------
        frame    : np.ndarray (H, W, 3) uint8 — POV RGB frame
        var_name : DIA variable the agent is currently targeting
        context  : "start" (beginning of skill) or "failed" (skill just failed)

        Returns
        -------
        MineRLVLMResult — cached if called again within min_interval_s
        """
        now = time.time()
        cached = self._cache.get(var_name)
        if cached is not None:
            ts, result = cached
            if now - ts < self._min_interval and result.error is None:
                return result

        if frame is None:
            return MineRLVLMResult(error="no frame")

        result = self._call_api(frame, var_name, context)
        self._cache[var_name] = (now, result)
        return result

    def stats(self) -> Dict[str, float]:
        return {
            "total_calls":   float(self._total_calls),
            "avg_latency_ms": self._total_latency / max(1, self._total_calls),
            "error_rate":    self._error_count / max(1, self._total_calls + self._error_count),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic  to use MineRLVLMAnalyzer")

        api_key = None
        for env_var in self.cfg.api_key_envs:
            api_key = os.environ.get(env_var)
            if api_key:
                break
        if not api_key:
            raise RuntimeError(
                f"No API key found. Set one of: {self.cfg.api_key_envs}"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _encode_frame(self, frame: np.ndarray) -> str:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("pip install Pillow  to use MineRLVLMAnalyzer")

        img = Image.fromarray(frame.astype(np.uint8))
        if self.cfg.frame_scale > 1:
            w, h = img.size
            img = img.resize((w * self.cfg.frame_scale, h * self.cfg.frame_scale),
                             Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_api(
        self,
        frame: np.ndarray,
        var_name: str,
        context: str,
    ) -> MineRLVLMResult:
        t0 = time.time()
        try:
            client = self._get_client()
            b64    = self._encode_frame(frame)

            skill_ctx  = _SKILL_CONTEXT.get(var_name, f"Goal: acquire {var_name}.")
            ctx_phrase = "before starting the skill" if context == "start" \
                         else "after the skill just FAILED — diagnose and suggest recovery"

            user_text = (
                f"The agent is targeting: {var_name.upper()}\n"
                f"Context: {ctx_phrase}\n"
                f"Skill detail: {skill_ctx}\n\n"
                "Analyze this Minecraft first-person view and call report_minecraft_scene "
                "with your assessment and a directive_actions list of 5-12 action indices."
            )

            response = client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=_SYSTEM_PROMPT,
                tools=[_TOOL_SCHEMA],
                tool_choice={"type": "any"},
                messages=[{
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
                        {"type": "text", "text": user_text},
                    ],
                }],
            )

            if isinstance(response, str):
                raise RuntimeError("API returned raw SSE string — check API key")

            # Parse tool_use block
            tool_input = None
            for block in response.content:
                if (getattr(block, "type", None) == "tool_use"
                        and block.name == "report_minecraft_scene"):
                    tool_input = block.input
                    break

            if tool_input is None:
                return MineRLVLMResult(
                    error="no tool_use block",
                    latency_ms=(time.time() - t0) * 1000,
                )

            # Build result
            latency = (time.time() - t0) * 1000
            self._total_calls   += 1
            self._total_latency += latency

            # Validate directive actions (clip to valid range)
            raw_actions = tool_input.get("directive_actions", [])
            if not isinstance(raw_actions, list):
                raw_actions = []
            directive = [int(a) for a in raw_actions if isinstance(a, (int, float))
                         and 0 <= int(a) <= 24]

            result = MineRLVLMResult(
                wood_visible           = float(np.clip(float(tool_input.get("wood_visible",           0)), 0, 1)),
                stone_visible          = float(np.clip(float(tool_input.get("stone_visible",          0)), 0, 1)),
                coal_visible           = float(np.clip(float(tool_input.get("coal_visible",           0)), 0, 1)),
                ironore_visible        = float(np.clip(float(tool_input.get("ironore_visible",        0)), 0, 1)),
                furnace_visible        = float(np.clip(float(tool_input.get("furnace_visible",        0)), 0, 1)),
                crafting_table_visible = float(np.clip(float(tool_input.get("crafting_table_visible", 0)), 0, 1)),
                open_space_ahead       = float(np.clip(float(tool_input.get("open_space_ahead",     0.5)), 0, 1)),
                underground            = float(np.clip(float(tool_input.get("underground",            0)), 0, 1)),
                directive_actions      = directive,
                reason                 = str(tool_input.get("reason", ""))[:200],
                latency_ms             = latency,
            )

            if self.cfg.verbose:
                print(f"  [VLM/{var_name}/{context}] {result.reason}  "
                      f"directive={result.directive_actions}  ({latency:.0f}ms)")

            return result

        except Exception as exc:
            self._error_count += 1
            latency = (time.time() - t0) * 1000
            err_msg = str(exc)
            if err_msg not in self._seen_errors:
                print(f"  [VLM/{var_name}] error: {err_msg}  ({latency:.0f}ms)")
                self._seen_errors.add(err_msg)
            return MineRLVLMResult(error=err_msg, latency_ms=latency)


# ── EVGS fusion helper ─────────────────────────────────────────────────────────

def fuse_with_inventory(
    inventory_vec: np.ndarray,
    vlm_result: MineRLVLMResult,
    visual_weight: float = 0.4,
) -> np.ndarray:
    """
    Fuse inventory-based binary EVGS vector with VLM visual presence estimates.

    Fusion rule:
      fused[i] = max(inventory[i],  visual_weight * visual[i])

    Inventory is authoritative for possession (binary 0/1).
    Visual signal boosts variables that are not yet acquired but *visible*.

    Parameters
    ----------
    inventory_vec  : np.ndarray [9] — binary {0,1} from evgs_minerl.py
    vlm_result     : MineRLVLMResult
    visual_weight  : float — scaling for visual signal (default 0.4 → max +0.4 boost)

    Returns
    -------
    np.ndarray [9] — fused values in [0, 1]
    """
    visual = vlm_result.visual_var_array()
    return np.maximum(inventory_vec.astype(float), visual_weight * visual)
