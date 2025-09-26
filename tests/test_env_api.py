import inspect

from dia.envs.base import EnvAPI


def test_env_api_has_expected_members():
    members = {
        "reset": inspect.isfunction,
        "step": inspect.isfunction,
        "action_space": lambda attr: isinstance(attr, property),
        "observation_space": lambda attr: isinstance(attr, property),
        "get_factors": inspect.isfunction,
        "intervene": inspect.isfunction,
    }

    for name, predicate in members.items():
        assert hasattr(EnvAPI, name), f"EnvAPI missing {name}"
        assert predicate(getattr(EnvAPI, name)), f"EnvAPI {name} has unexpected type"


def test_env_api_defaults_implemented():
    class DummyEnv(EnvAPI):
        def __init__(self):
            self._action_space = "discrete"
            self._observation_space = "vector"

        def reset(self, seed=None, factors=None):  # pragma: no cover - trivial glue
            return "obs"

        def step(self, action):  # pragma: no cover - trivial glue
            return "obs", 1.0, False, {}

        @property
        def action_space(self):
            return self._action_space

        @property
        def observation_space(self):
            return self._observation_space

    env = DummyEnv()
    assert env.get_factors() == {}
    assert env.intervene({"foo": 1}) is False
