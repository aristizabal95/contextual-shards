"""ProcgenMazeEnv — wraps procgen-tools maze venv.

Requires procgen + procgen-tools. Uses create_venv from procgen_tools.maze
for the CHW observation format expected by the IMPALA model.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.environment_module import register_env
from src.environment_module.base_env import BaseEnv


@register_env("maze")
class ProcgenMazeEnv(BaseEnv):
    """Wraps procgen-tools create_venv for the maze environment.

    Observations are (C, H, W) float32 numpy arrays (single env, batch dim removed).
    Positions are (row, col) in inner grid coordinates.

    Args:
        cfg: Config with cfg.environment.{num_levels, seed}
    """

    def __init__(self, cfg: Any) -> None:
        try:
            from procgen_tools.maze import create_venv  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "procgen-tools not installed. ProcgenMazeEnv requires procgen + procgen-tools."
            ) from e

        env_cfg = cfg.environment
        self._venv = create_venv(
            num=1,
            start_level=env_cfg.seed,
            num_levels=env_cfg.num_levels,
        )
        self._obs: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        obs = self._venv.reset()  # (1, C, H, W)
        self._obs = obs[0]
        return self._obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, rew, done, _info = self._venv.step(np.array([action]))
        self._obs = obs[0]
        return self._obs, float(rew[0]), bool(done[0]), {}

    def cheese_pos(self) -> Tuple[int, int]:
        """Return (row, col) of cheese in the inner grid."""
        return self._get_object_pos(2)  # CHEESE = 2

    def agent_pos(self) -> Tuple[int, int]:
        """Return (row, col) of agent (mouse) in the inner grid."""
        return self._get_object_pos(25)  # MOUSE = 25

    def _get_object_pos(self, obj_id: int) -> Tuple[int, int]:
        try:
            from procgen_tools.maze import state_from_venv  # type: ignore[import]
        except ImportError:
            return (0, 0)
        state = state_from_venv(self._venv, 0)
        grid = state.inner_grid()
        positions = np.argwhere(grid == obj_id)
        if len(positions) == 0:
            return (0, 0)
        return int(positions[0][0]), int(positions[0][1])

    def get_obs_no_cheese(self) -> np.ndarray:
        """Return an observation identical to the current state but with cheese removed.

        Maze layout and agent position are preserved; only the cheese cell is set to
        EMPTY. This produces a minimal, low-noise corrupted observation suitable for
        causal tracing experiments.
        """
        try:
            from procgen_tools import maze  # type: ignore[import]
        except ImportError as e:
            raise ImportError("procgen-tools required for cheese manipulation.") from e

        tmp_venv = maze.copy_venv(self._venv, 0)
        maze.remove_cheese(tmp_venv, 0)
        obs = tmp_venv.reset()  # (1, C, H, W) — reset respects prior set_state call
        return obs[0].astype(np.float32)

    @property
    def venv(self) -> Any:
        """Expose the underlying venv for procgen-tools utilities."""
        return self._venv

    def close(self) -> None:
        self._venv.close()
