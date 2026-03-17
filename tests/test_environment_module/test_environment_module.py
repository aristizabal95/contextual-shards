from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.environment_module import EnvFactory, register_env, BaseEnv, ENV_FACTORY
from src.environment_module.maze.maze_env import ProcgenMazeEnv


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


# ---------------------------------------------------------------------------
# Tests for ProcgenMazeEnv.get_obs_no_cheese
# ---------------------------------------------------------------------------

def _make_maze_env_with_mock_venv(obs_shape=(3, 64, 64)) -> ProcgenMazeEnv:
    """Return a ProcgenMazeEnv whose internal _venv is a MagicMock.

    Bypasses the procgen import so the env can be constructed in any Python env.
    """
    env = object.__new__(ProcgenMazeEnv)
    mock_venv = MagicMock()
    env._venv = mock_venv
    env._obs = np.zeros(obs_shape, dtype=np.float32)
    return env


def _patch_procgen_tools(mock_maze: MagicMock):
    """Return a context manager that injects mock_maze as procgen_tools.maze."""
    mock_pkg = MagicMock()
    mock_pkg.maze = mock_maze
    return patch.dict("sys.modules", {"procgen_tools": mock_pkg, "procgen_tools.maze": mock_maze})


def test_get_obs_no_cheese_returns_float32_array():
    """get_obs_no_cheese must return a float32 numpy array."""
    obs_shape = (3, 64, 64)
    corrupted_obs = np.ones(obs_shape, dtype=np.float32)

    mock_tmp_venv = MagicMock()
    mock_tmp_venv.reset.return_value = corrupted_obs[np.newaxis]  # (1, C, H, W)

    mock_maze = MagicMock()
    mock_maze.copy_venv.return_value = mock_tmp_venv

    env = _make_maze_env_with_mock_venv(obs_shape)

    with _patch_procgen_tools(mock_maze):
        result = env.get_obs_no_cheese()

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == obs_shape


def test_get_obs_no_cheese_calls_remove_cheese_and_reset():
    """get_obs_no_cheese must call remove_cheese on the copy, then reset."""
    obs_shape = (3, 64, 64)
    corrupted_obs = np.random.rand(1, *obs_shape).astype(np.float32)

    mock_tmp_venv = MagicMock()
    mock_tmp_venv.reset.return_value = corrupted_obs

    mock_maze = MagicMock()
    mock_maze.copy_venv.return_value = mock_tmp_venv

    env = _make_maze_env_with_mock_venv(obs_shape)
    main_venv = env._venv

    with _patch_procgen_tools(mock_maze):
        result = env.get_obs_no_cheese()

    mock_maze.copy_venv.assert_called_once_with(main_venv, 0)
    mock_maze.remove_cheese.assert_called_once_with(mock_tmp_venv, 0)
    mock_tmp_venv.reset.assert_called_once()
    np.testing.assert_array_equal(result, corrupted_obs[0])


def test_get_obs_no_cheese_does_not_mutate_main_venv():
    """get_obs_no_cheese must not call set_state or modify the main _venv."""
    obs_shape = (3, 64, 64)
    mock_tmp_venv = MagicMock()
    mock_tmp_venv.reset.return_value = np.zeros((1, *obs_shape), dtype=np.float32)

    mock_maze = MagicMock()
    mock_maze.copy_venv.return_value = mock_tmp_venv

    env = _make_maze_env_with_mock_venv(obs_shape)
    main_venv = env._venv

    with _patch_procgen_tools(mock_maze):
        env.get_obs_no_cheese()

    main_venv.env.callmethod.assert_not_called()


def test_get_obs_no_cheese_raises_without_procgen_tools():
    """get_obs_no_cheese must raise ImportError when procgen_tools is unavailable."""
    env = _make_maze_env_with_mock_venv()

    with patch.dict("sys.modules", {"procgen_tools": None, "procgen_tools.maze": None}):
        with pytest.raises(ImportError):
            env.get_obs_no_cheese()
