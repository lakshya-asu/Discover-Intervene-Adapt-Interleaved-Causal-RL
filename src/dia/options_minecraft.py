# src/dia/options_minecraft.py
from __future__ import annotations

from typing import Any, Dict, Optional

from .types import Subgoal
from .evgs import EVGS
from .options import OptionPolicy, OptionConfig
from .envs.minecraft2d import ChainConfig  # thresholds


# One-shot action mapping (kept for reference/tests)
_VARIDX_TO_ACTION = {
    0: 0,  # wood -> gather wood
    1: 1,  # stone -> gather stone
    2: 2,  # coal -> gather coal
    3: 3,  # ironore -> mine iron ore
    4: 4,  # furnace -> craft furnace
    5: 5,  # stonepickaxe -> craft stone pickaxe
    6: 6,  # iron -> smelt iron
    7: 7,  # ironpickaxe -> craft iron pickaxe
    8: 8,  # diamond -> mine diamond
}


class RuleOption(OptionPolicy):
    """Minimal one-action option (legacy)."""
    def __init__(self, subgoal: Subgoal, cfg: OptionConfig, action_space=None):
        super().__init__(subgoal, cfg)
        self._action = int(_VARIDX_TO_ACTION.get(self.subgoal.var_index, 9))
    def act(self, obs):
        return self._action


class ResourceOption(OptionPolicy):
    """
    Heuristic, resource-aware option that sequences primitive actions to satisfy
    prerequisites for the subgoal (e.g., gather stone before crafting a furnace).
    """

    def __init__(self, subgoal: Subgoal, cfg: OptionConfig):
        super().__init__(subgoal, cfg)

    # ---- micro policy ----

    @staticmethod
    def _cfg_from_env(env) -> ChainConfig:
        return getattr(env, "cfg", ChainConfig())

    def _next_action(self, x, env) -> int:
        """
        Choose the next primitive action to move toward achieving subgoal.
        x indexes (by VAR_NAMES in env): [wood, stone, coal, ironore, furnace, stonepickaxe, iron, ironpickaxe, diamond]
        """
        j = self.subgoal.var_index
        cfg = self._cfg_from_env(env)

        wood, stone, coal, ironore, furnace, stonepick, iron, ironpick, diamond = x.tolist()

        # helpers
        def need_wood(n): return wood < n
        def need_stone(n): return stone < n

        if j == 0:  # wood
            return 0
        if j == 1:  # stone
            return 1
        if j == 2:  # coal
            return 2

        if j == 4:  # furnace
            if furnace >= 1:
                return 9
            if stone >= cfg.stone_for_furnace:
                return 4
            return 1  # gather stone

        if j == 5:  # stone pickaxe
            if stonepick >= 1:
                return 9
            if wood >= cfg.wood_for_stone_pick and stone >= cfg.stone_for_stone_pick:
                return 5
            # gather missing
            if need_wood(cfg.wood_for_stone_pick):
                return 0
            return 1

        if j == 3:  # iron ore (needs a pickaxe)
            if stonepick >= 1 or ironpick >= 1:
                return 3  # mine ore
            # craft stone pickaxe path
            if wood >= cfg.wood_for_stone_pick and stone >= cfg.stone_for_stone_pick:
                return 5
            if need_wood(cfg.wood_for_stone_pick):
                return 0
            return 1

        if j == 6:  # iron (needs furnace + ore + coal)
            if furnace < 1:
                # build furnace path
                if stone >= cfg.stone_for_furnace:
                    return 4
                return 1
            # ensure materials
            if ironore < 1:
                # mine ore (ensure pickaxe)
                if stonepick >= 1 or ironpick >= 1:
                    return 3
                # craft stone pickaxe
                if wood >= cfg.wood_for_stone_pick and stone >= cfg.stone_for_stone_pick:
                    return 5
                if need_wood(cfg.wood_for_stone_pick):
                    return 0
                return 1
            if coal < 1:
                return 2
            return 6  # smelt

        if j == 7:  # iron pickaxe (needs iron*3 + wood*2)
            if ironpick >= 1:
                return 9
            if iron >= cfg.iron_for_iron_pick and wood >= cfg.wood_for_iron_pick:
                return 7
            # ensure iron total
            if iron < cfg.iron_for_iron_pick:
                # recursively produce iron (smelt path)
                if furnace < 1:
                    if stone >= cfg.stone_for_furnace:
                        return 4
                    return 1
                if ironore < 1:
                    if stonepick >= 1 or ironpick >= 1:
                        return 3
                    if wood >= cfg.wood_for_stone_pick and stone >= cfg.stone_for_stone_pick:
                        return 5
                    if need_wood(cfg.wood_for_stone_pick):
                        return 0
                    return 1
                if coal < 1:
                    return 2
                return 6
            # ensure wood
            if need_wood(cfg.wood_for_iron_pick):
                return 0
            return 7

        if j == 8:  # diamond (needs iron pickaxe)
            if ironpick >= 1:
                return 8
            # craft iron pick first
            return self._next_action(x, env.__class__(cfg=self._cfg_from_env(env))) if False else (
                # fallback to basic iron-pick crafting steps:
                7 if (iron >= cfg.iron_for_iron_pick and wood >= cfg.wood_for_iron_pick) else
                (0 if wood < cfg.wood_for_iron_pick else 6)
            )

        # default: noop
        return 9

    # ---- run loop ----

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        obs = env.get_obs() if hasattr(env, "get_obs") else env.reset()
        x_prev = evgs.extract(obs)
        steps = 0
        success = False
        trajectory = []
        while steps < self.cfg.max_steps:
            x = evgs.extract(obs)
            # choose next primitive action
            action = self._next_action(x, env)
            next_obs, ext_rew, done, info = env.step(action)
            x_next = evgs.extract(next_obs)
            succ_this = EVGS.predicate_holds(x, x_next, self.subgoal)
            trajectory.append((obs, action, next_obs, succ_this))
            obs = next_obs
            steps += 1
            if succ_this:
                success = True
                if self.cfg.terminate_on_success:
                    break
            if done and not (isinstance(info, dict) and info.get("soft_continue", False)):
                break
        return {"success": success, "steps": steps, "trajectory": trajectory, "final_obs": obs}
