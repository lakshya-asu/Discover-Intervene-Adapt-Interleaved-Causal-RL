# src/dia/evgs_montezuma.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np

try:
    import gym
except Exception:
    gym = None

from .evgs_adapters import make_montezuma_evgs, InfoObsWrapper


@dataclass
class MontezumaDetectorsConfig:
    room_addr: Optional[int] = None
    has_key_addr: Optional[int] = None
    door_addr: Optional[int] = None
    player_x_addr: Optional[int] = None
    player_y_addr: Optional[int] = None
    skull_x_addr: Optional[int] = None
    skull_y_addr: Optional[int] = None
    has_key_values: Optional[List[int]] = None
    door_open_values: Optional[List[int]] = None
    door_threshold: Optional[int] = None
    near_radius: float = 8.0
    yellow_r_min: int = 180
    yellow_g_min: int = 180
    yellow_b_max: int = 80
    yellow_count_threshold: int = 30


class MontezumaRAMDetector:
    def __init__(self, cfg: MontezumaDetectorsConfig):
        self.cfg = cfg

    @staticmethod
    def _safe_ram(ram: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if ram is None:
            return None
        ram = np.array(ram, dtype=np.uint8).reshape(-1)
        return ram if ram.ndim == 1 and ram.size >= 64 else None

    def _byte(self, ram: np.ndarray, addr: Optional[int]) -> Optional[int]:
        if addr is None:
            return None
        if 0 <= addr < ram.shape[0]:
            return int(ram[addr])
        return None

    def extract_from_ram(self, ram: Optional[np.ndarray]) -> Dict[str, float]:
        out = {"has_key": 0.0, "door_open": 0.0, "room_id": 0.0, "skull_near": 0.0}
        ram = self._safe_ram(ram)
        if ram is None:
            return out
        room_byte = self._byte(ram, self.cfg.room_addr)
        if room_byte is not None:
            out["room_id"] = float(room_byte)
        key_byte = self._byte(ram, self.cfg.has_key_addr)
        if key_byte is not None:
            if self.cfg.has_key_values is not None:
                out["has_key"] = 1.0 if key_byte in self.cfg.has_key_values else 0.0
            else:
                out["has_key"] = 1.0 if key_byte > 0 else 0.0
        door_byte = self._byte(ram, self.cfg.door_addr)
        if door_byte is not None:
            if self.cfg.door_open_values is not None:
                out["door_open"] = 1.0 if door_byte in self.cfg.door_open_values else 0.0
            elif self.cfg.door_threshold is not None:
                out["door_open"] = 1.0 if door_byte >= self.cfg.door_threshold else 0.0
            else:
                out["door_open"] = 1.0 if door_byte > 0 else 0.0
        px = self._byte(ram, self.cfg.player_x_addr)
        py = self._byte(ram, self.cfg.player_y_addr)
        ex = self._byte(ram, self.cfg.skull_x_addr)
        ey = self._byte(ram, self.cfg.skull_y_addr)
        if None not in (px, py, ex, ey):
            dist = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
            out["skull_near"] = 1.0 if dist <= self.cfg.near_radius else 0.0
        return out


class MontezumaPixelDetector:
    def __init__(self, yellow_r_min=180, yellow_g_min=180, yellow_b_max=80, count_threshold=30):
        self.r_min = int(yellow_r_min)
        self.g_min = int(yellow_g_min)
        self.b_max = int(yellow_b_max)
        self.count_threshold = int(count_threshold)

    def key_present(self, frame: Optional[np.ndarray]) -> bool:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] != 3:
            return False
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        mask = (r >= self.r_min) & (g >= self.g_min) & (b <= self.b_max)
        return bool(np.count_nonzero(mask) >= self.count_threshold)


class MontezumaInfoWrapper(gym.Wrapper if gym else object):
    def __init__(self, env, cfg: MontezumaDetectorsConfig):
        if gym:
            super().__init__(env)
        self.cfg = cfg
        self.ram_det = MontezumaRAMDetector(cfg)
        self.pix_det = MontezumaPixelDetector(cfg.yellow_r_min, cfg.yellow_g_min, cfg.yellow_b_max, cfg.yellow_count_threshold)
        self._prev_key_present = None
        self._has_key_state = 0.0
        self._room_state = 0.0
        self._last_obs = None

    def _get_ram(self) -> Optional[np.ndarray]:
        try:
            ale = getattr(self.env.unwrapped, "ale", None)
            return None if ale is None else np.array(ale.getRAM(), dtype=np.uint8).reshape(-1)
        except Exception:
            return None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_key_present = None
        self._has_key_state = 0.0
        ram = self._get_ram()
        if ram is not None:
            self._room_state = float(self.ram_det.extract_from_ram(ram).get("room_id", 0.0))
        info = {"has_key": self._has_key_state, "door_open": 0.0, "room_id": self._room_state, "skull_near": 0.0}
        self._last_obs = {"obs": obs, "info": info}
        return self._last_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        injected = {"has_key": self._has_key_state, "door_open": 0.0, "room_id": self._room_state, "skull_near": 0.0}
        ram = self._get_ram()
        if ram is not None:
            sig = self.ram_det.extract_from_ram(ram)
            if "room_id" in sig:
                self._room_state = float(sig["room_id"])
            if sig.get("has_key", 0.0) > 0.0:
                self._has_key_state = 1.0
            injected.update(sig)
        frame = obs if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[-1] == 3 else None
        if self._has_key_state <= 0.0 and frame is not None:
            key_now = self.pix_det.key_present(frame)
            if self._prev_key_present is None:
                self._prev_key_present = key_now
            elif self._prev_key_present is True and key_now is False:
                self._has_key_state = 1.0
            self._prev_key_present = key_now
        injected["has_key"] = self._has_key_state
        injected["room_id"] = self._room_state
        info = dict(info or {})
        info.update(injected)
        info.setdefault("soft_continue", True)
        self._last_obs = {"obs": obs, "info": info}
        return self._last_obs, rew, done, info

    # NEW
    def get_obs(self):
        if self._last_obs is None:
            return self.reset()
        return self._last_obs


def wrap_montezuma_env(env, cfg: Optional[MontezumaDetectorsConfig] = None, return_evgs: bool = False):
    cfg = cfg or MontezumaDetectorsConfig()
    env = InfoObsWrapper(env)
    env = MontezumaInfoWrapper(env, cfg)
    if return_evgs:
        evgs = make_montezuma_evgs()
        return env, evgs
    return env


# ---------------------------------------------------------------------------
# Rich Montezuma EVGS — 8 semantic variables with player-zone awareness
# ---------------------------------------------------------------------------
#
# Known Atari 2600 RAM layout for Montezuma's Revenge (ALE / gym):
#   Byte 42  – player X position  (0 = left edge, ~152 = right edge)
#   Byte 43  – player Y position  (lower byte value = HIGHER on screen,
#              e.g. 148 ≈ top platform, 235 ≈ bottom floor)
#   Byte  3  – current room index (value 1 = first room in this ALE version)
#   Byte 47  – skull/rolling-enemy X (oscillates 28–58, validated by live probe)
#              Skull Y is fixed at ~145 in room 1 (rolls on mid platform only)
#
# Score detection: use per-step env reward (key = +100, door = +300) rather
# than BCD score parsing — more reliable across ALE versions.
#
# These addresses were validated by live ALE probing (ale-py 0.11.2,
# gymnasium 1.2.3).  Minor offsets (±1 byte) possible between ROM versions;
# zone thresholds below were chosen conservatively.
#
# Room-1 spatial zones (approximate pixel coords mapped to RAM coords):
#   Key platform (top-right):  X ∈ [100, 155], Y ∈ [140, 168]
#   Door (bottom-left):        X ∈ [ 10,  58], Y ∈ [215, 250]
#   Central rope:              X ∈ [ 65,  95], Y ∈ [165, 215]
#   Upper level (any):         Y < 175

_MONTE_RAM_PLAYER_X  = 42
_MONTE_RAM_PLAYER_Y  = 43
_MONTE_RAM_ROOM      = 3
_MONTE_RAM_SKULL_X   = 47   # validated: oscillates 28-58 each NOOP step
_SKULL_Y_ROOM1       = 145  # fixed Y for skull's mid-platform in room 1
_SKULL_NEAR_RADIUS   = 25.0

_ROOM1_KEY_X       = (100, 155)
_ROOM1_KEY_Y       = (140, 168)
_ROOM1_DOOR_X      = (10,   58)
_ROOM1_DOOR_Y      = (215, 250)
_ROOM1_ROPE_X      = (55,  102)    # widened: probe shows X=63 (LEFT ladder) to X=89 (RIGHT)
_ROOM1_ROPE_Y      = (160, 220)    # slightly widened vertically
_ROOM1_UPPER_Y_MAX = 185          # Y < this → on upper/mid platform (incl. rope platform)


def _bcd_byte(b: int) -> int:
    """Convert one BCD-encoded byte to its decimal integer value."""
    return ((b >> 4) & 0xF) * 10 + (b & 0xF)


def _in_zone(x, y, x_range, y_range) -> float:
    return 1.0 if (x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]) else 0.0


class MontezumaRichWrapper:
    """
    Wraps a raw Atari gym/gymnasium environment for Montezuma's Revenge and
    injects eight semantically rich variables into the info dict:

      has_key         – agent has collected the key (latches to 1 once gained)
      door_open       – agent has passed through a locked door
      at_key_zone     – player currently in the top-right key-platform region
      at_door_zone    – player currently in the bottom-left door region
      on_upper_level  – player is on any upper platform (Y < threshold)
      near_rope       – player is on/adjacent to the central hanging rope
      skull_near      – skull enemy is within proximity radius
      score_gained    – score increased this step (proxy for any progress event)

    Detection strategy:
      - Player/enemy positions: directly from ALE RAM bytes.
      - has_key / door_open: latches triggered by score-threshold changes
        (key pickup = +100 pts, door traversal = +300 pts in Room 1).
      - Zone variables: derived from player position vs. fixed spatial bounds.

    Exposes the standard gym step/reset interface plus get_obs() so it can
    be used directly with DIARunner without an additional Info wrapper.
    """

    # 8 variable names in canonical order (must match make_montezuma_evgs_rich)
    VAR_NAMES: List[str] = [
        "has_key", "door_open",
        "at_key_zone", "at_door_zone",
        "on_upper_level", "near_rope",
        "skull_near", "score_gained",
    ]

    def __init__(self, env):
        self.env = env
        self._player_x  = 77.0
        self._player_y  = 235.0
        self._has_key   = 0.0
        self._door_open = 0.0
        self._room      = 0.0
        self._last_obs_dict: Optional[dict] = None
        # Frame recording support
        self._recording: bool = False
        self._frames: List = []

    # ----------------------------------------------------------
    # Internal helpers

    def _get_ram(self) -> Optional[np.ndarray]:
        """Walk the env wrapper stack to find the ALE object and fetch RAM."""
        # Walk the .env chain (covers most wrapper stacks)
        inner = self.env
        while inner is not None:
            ale = getattr(inner, "ale", None)
            if ale is not None:
                try:
                    return np.asarray(ale.getRAM(), dtype=np.uint8).reshape(-1)
                except Exception:
                    return None
            inner = getattr(inner, "env", None)
        # Fallback: try .unwrapped (gymnasium OrderEnforcing/PassiveEnvChecker pattern)
        try:
            unwrapped = self.env.unwrapped
            ale = getattr(unwrapped, "ale", None)
            if ale is not None:
                return np.asarray(ale.getRAM(), dtype=np.uint8).reshape(-1)
        except Exception:
            pass
        return None

    def _build_info(self, ram: Optional[np.ndarray], score_gained: bool) -> dict:
        info: dict = {
            "has_key":        self._has_key,
            "door_open":      self._door_open,
            "at_key_zone":    0.0,
            "at_door_zone":   0.0,
            "on_upper_level": 0.0,
            "near_rope":      0.0,
            "skull_near":     0.0,
            "score_gained":   1.0 if score_gained else 0.0,
            "room_id":        self._room,
            "soft_continue":  True,
        }
        if ram is not None and len(ram) > max(_MONTE_RAM_SKULL_X, _MONTE_RAM_ROOM) + 1:
            px = float(ram[_MONTE_RAM_PLAYER_X])
            py = float(ram[_MONTE_RAM_PLAYER_Y])
            self._player_x = px
            self._player_y = py
            self._room     = float(ram[_MONTE_RAM_ROOM])

            info["at_key_zone"]    = _in_zone(px, py, _ROOM1_KEY_X,  _ROOM1_KEY_Y)
            info["at_door_zone"]   = _in_zone(px, py, _ROOM1_DOOR_X, _ROOM1_DOOR_Y)
            info["on_upper_level"] = 1.0 if py < _ROOM1_UPPER_Y_MAX else 0.0
            info["near_rope"]      = _in_zone(px, py, _ROOM1_ROPE_X, _ROOM1_ROPE_Y)
            info["room_id"]        = self._room

            sx   = float(ram[_MONTE_RAM_SKULL_X])
            dist = ((px - sx) ** 2 + (py - _SKULL_Y_ROOM1) ** 2) ** 0.5
            info["skull_near"] = 1.0 if dist < _SKULL_NEAR_RADIUS else 0.0
        return info

    def _update_score_latches(self, reward: float) -> bool:
        """
        Use per-step env reward to latch has_key / door_open.
        Key pickup  = +100 reward in room 1.
        Door traversal = +300 reward.
        Returns True if any reward was gained this step.
        """
        gained = reward > 0.0
        if reward >= 100.0:
            self._has_key = 1.0
        if reward >= 300.0:
            self._door_open = 1.0
        return gained

    def _pack(self, raw_obs, info: dict) -> dict:
        return {"obs": raw_obs, "info": info}

    # ----------------------------------------------------------
    # Gym interface

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # gymnasium returns (obs, info); legacy gym returns obs or dict
        if isinstance(result, tuple) and len(result) == 2:
            raw_obs, _env_info = result
        elif isinstance(result, dict):
            raw_obs = result.get("obs", result)
        else:
            raw_obs = result
        # Reset latches
        self._has_key   = 0.0
        self._door_open = 0.0
        ram = self._get_ram()
        info = self._build_info(ram, score_gained=False)
        packed = self._pack(raw_obs, info)
        self._last_obs_dict = packed
        if self._recording:
            frame = self._render_frame()
            if frame is not None:
                self._frames = [frame]
        return packed

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            raw_step, rew, terminated, truncated, env_info = result
            done = bool(terminated or truncated)
        elif isinstance(result, tuple) and len(result) == 4:
            raw_step, rew, done, env_info = result
        else:
            raise ValueError(f"Unexpected step() return: {type(result)}")

        raw_obs = raw_step["obs"] if isinstance(raw_step, dict) else raw_step

        ram = self._get_ram()
        score_gained = self._update_score_latches(float(rew))
        info = self._build_info(ram, score_gained)

        if isinstance(env_info, dict):
            for k, v in env_info.items():
                info.setdefault(k, v)

        packed = self._pack(raw_obs, info)
        self._last_obs_dict = packed

        if self._recording:
            frame = self._render_frame()
            if frame is not None:
                self._frames.append(frame)

        return packed, float(rew), done, info

    def get_obs(self) -> dict:
        """Return last observation without resetting (for DIARunner compatibility)."""
        if self._last_obs_dict is None:
            return self.reset()
        return self._last_obs_dict

    # ----------------------------------------------------------
    # Frame recording

    def _render_frame(self):
        try:
            frame = self.env.render()
            return frame if frame is not None else None
        except Exception:
            return None

    def start_recording(self):
        """Begin accumulating rendered frames from subsequent steps."""
        self._recording = True
        self._frames = []

    def stop_recording(self):
        """Stop accumulating frames."""
        self._recording = False

    def get_frames(self) -> list:
        """Return a copy of the accumulated frame list."""
        return list(self._frames)

    # ----------------------------------------------------------
    # Proxy attributes so the wrapper behaves like the inner env

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def unwrapped(self):
        inner = self.env
        while hasattr(inner, "unwrapped"):
            inner = inner.unwrapped
        return inner

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_montezuma_evgs_rich() -> "EVGS":  # type: ignore[name-defined]
    """
    EVGS for the 8-variable rich Montezuma representation produced by
    MontezumaRichWrapper.

    Variable order matches MontezumaRichWrapper.VAR_NAMES:
      0  has_key          – [0, 1] latching binary
      1  door_open        – [0, 1] latching binary
      2  at_key_zone      – [0, 1] spatial indicator (top-right platform)
      3  at_door_zone     – [0, 1] spatial indicator (bottom-left door)
      4  on_upper_level   – [0, 1] spatial indicator (any upper platform)
      5  near_rope        – [0, 1] spatial indicator (central rope)
      6  skull_near       – [0, 1] enemy proximity
      7  score_gained     – [0, 1] momentary progress signal
    """
    from .evgs import EVGS

    names = MontezumaRichWrapper.VAR_NAMES

    def obs_to_vars(obs) -> np.ndarray:
        if isinstance(obs, dict):
            info = obs.get("info", {}) or {}
        else:
            info = {}
        return np.array([
            float(bool(info.get("has_key",        0))),
            float(bool(info.get("door_open",       0))),
            float(bool(info.get("at_key_zone",     0))),
            float(bool(info.get("at_door_zone",    0))),
            float(bool(info.get("on_upper_level",  0))),
            float(bool(info.get("near_rope",       0))),
            float(bool(info.get("skull_near",      0))),
            float(bool(info.get("score_gained",    0))),
        ], dtype=float)

    return EVGS(var_names=names, obs_to_vars=obs_to_vars)
