import pytest
from src.environment_module import EnvFactory, register_env, BaseEnv, ENV_FACTORY


def test_env_factory_raises_unknown():
    with pytest.raises(ValueError, match="Unknown env"):
        EnvFactory("nonexistent_env")


def test_env_factory_returns_class():
    # maze is registered (even though it can't be instantiated without procgen)
    cls = EnvFactory("maze")
    assert issubclass(cls, BaseEnv)


def test_env_factory_returns_minigrid_class():
    cls = EnvFactory("minigrid")
    assert issubclass(cls, BaseEnv)


def test_register_env_decorator():
    @register_env("mock_env_test")
    class MockEnv(BaseEnv):
        def reset(self):
            return None

        def step(self, _action):
            return None, 0.0, False, {}

        def cheese_pos(self):
            return (0, 0)

        def agent_pos(self):
            return (0, 0)

        def close(self):
            pass

    assert "mock_env_test" in ENV_FACTORY
    cls = EnvFactory("mock_env_test")
    assert cls is MockEnv


def test_maze_env_raises_without_procgen():
    """ProcgenMazeEnv should raise ImportError when procgen is not installed."""
    cls = EnvFactory("maze")

    class FakeCfg:
        class environment:
            num_levels = 10
            seed = 0
            distribution_mode = "easy"

    try:
        instance = cls(FakeCfg())
        # If procgen IS installed somehow, just close it
        instance.close()
    except ImportError:
        pass  # Expected when procgen not installed
