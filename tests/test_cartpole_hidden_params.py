import numpy as np
import pytest

from dia.envs.cartpole_hp import CartPoleHiddenParamsEnv


def test_cartpole_hidden_params_vary_with_seed():
    try:
        env = CartPoleHiddenParamsEnv()
    except ImportError as exc:
        pytest.skip(str(exc))

    try:
        obs, info = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        factors_seed_1 = env.get_factors()

        obs, info = env.reset(seed=2)
        factors_seed_2 = env.get_factors()

        assert factors_seed_1 != factors_seed_2
        assert factors_seed_1["pole_length"] != factors_seed_2["pole_length"] or \
            factors_seed_1["pole_mass"] != factors_seed_2["pole_mass"]
    finally:
        env.close()
