import numpy as np
import pytest

from dia.envs.procgen_coinrun import ProcgenCoinRunEnv


def test_procgen_coinrun_wrapper_smoke():
    try:
        env = ProcgenCoinRunEnv(start_level=0, num_levels=0)
    except ImportError as exc:
        pytest.skip(str(exc))

    try:
        obs, info = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.uint8
        assert obs.shape == (64, 64, 3)
        assert isinstance(info, dict)

        assert env.action_space.n == env.NUM_ACTIONS
        assert env.level_seed is None or isinstance(env.level_seed, int)

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.dtype == np.uint8
            assert obs.shape == (64, 64, 3)
            assert isinstance(info, dict)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
    finally:
        env.close()
