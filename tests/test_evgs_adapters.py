# tests/test_evgs_adapters.py
from __future__ import annotations

import numpy as np
from dia.evgs_adapters import make_montezuma_evgs, make_coinrun_evgs, make_causalworld_evgs

def test_montezuma_adapter():
    evgs = make_montezuma_evgs()
    obs = {"obs": np.zeros((4,)), "info": {"has_key": 1, "door_open": 0, "room_id": 7, "skull_near": 1}}
    x = evgs.extract(obs)
    assert x.shape[0] == 4
    assert x[0] == 1.0 and x[1] == 0.0 and abs(x[2] - 0.07) < 1e-6 and x[3] == 1.0

def test_coinrun_adapter():
    evgs = make_coinrun_evgs()
    obs = {"obs": np.zeros((3,)), "info": {"coin_collected": 1, "progress": 0.42, "enemy_near": 0}}
    x = evgs.extract(obs)
    assert (x == np.array([1.0, 0.42, 0.0])).all()

def test_causalworld_adapter():
    evgs = make_causalworld_evgs()
    obs = {"obs": np.zeros((3,)), "info": {"stack_height": 2, "distance_to_goal": 0.3, "grasped": 1}}
    x = evgs.extract(obs)
    assert (x == np.array([2.0, 0.3, 1.0])).all()
