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
