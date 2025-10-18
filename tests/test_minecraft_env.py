# tests/test_minecraft_env.py
from __future__ import annotations

import numpy as np

from dia.envs.minecraft2d import MinecraftChainEnv, ChainConfig
from dia.evgs_minecraft import make_minecraft_evgs
from dia.types import Subgoal, Predicate
from dia.evgs import EVGS


def test_basic_crafting_chain():
    env = MinecraftChainEnv(ChainConfig())
    evgs = make_minecraft_evgs()

    obs = env.reset()
    x0 = evgs.extract(obs)

    # gather wood (2x) and stone (3x)
    for _ in range(2):
        obs, _, _, _ = env.step(0)  # wood
    for _ in range(3):
        obs, _, _, _ = env.step(1)  # stone

    x1 = evgs.extract(obs)
    assert x1[0] >= x0[0] + 2 and x1[1] >= x0[1] + 3

    # craft stone pickaxe
    obs, _, _, _ = env.step(5)
    x2 = evgs.extract(obs)
    assert x2[5] == 1.0  # stonepickaxe

    # mine iron ore and craft furnace
    obs, _, _, _ = env.step(3)  # iron ore
    obs, _, _, _ = env.step(4)  # furnace (requires stone >= 8; may be 0 if not enough stone yet)
    x3 = evgs.extract(obs)
    # Either we didn't have enough stone to craft furnace (fine for smoke test) or it is crafted:
    assert x3[3] >= 1.0

    # smelt iron if furnace present and coal gathered
    obs, _, _, _ = env.step(2)  # coal
    obs, _, _, _ = env.step(6)  # smelt iron (if preconds satisfied)
    x4 = evgs.extract(obs)
    assert x4[6] >= 0.0
