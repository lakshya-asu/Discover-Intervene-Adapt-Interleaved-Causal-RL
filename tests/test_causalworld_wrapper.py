import numpy as np
import pytest

from dia.envs.causalworld_env import CausalWorldPushingEnv


def test_causalworld_wrapper_smoke():
    try:
        env = CausalWorldPushingEnv()
    except ImportError as exc:
        pytest.skip(str(exc))

    try:
        obs, info = env.reset(seed=123)
        assert "rgb" in obs
        assert isinstance(obs["rgb"], np.ndarray)
        assert obs["rgb"].shape == (64, 64, 3)
        assert obs["rgb"].dtype == np.uint8

        if "proprio" in obs:
            assert isinstance(obs["proprio"], np.ndarray)
            assert obs["proprio"].dtype == np.float32

        assert "factors" in info
        assert isinstance(info["factors"], dict)
        assert isinstance(env.get_factors(), dict)

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert obs["rgb"].shape == (64, 64, 3)
            assert isinstance(info, dict)
    finally:
        env.close()
