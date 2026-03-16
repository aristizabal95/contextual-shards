"""ProcgenMazeEnv — wraps procgen-tools maze environment.

Requires procgen + procgen-tools (Python 3.10). If not installed,
the class is still importable but instantiation will raise ImportError.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.environment_module import register_env
from src.environment_module.base_env import BaseEnv


@register_env("maze")
class ProcgenMazeEnv(BaseEnv):
    """Thin wrapper around procgen-tools venv for the maze environment.

    Args:
        cfg: Config object with cfg.environment.{num_levels, seed, distribution_mode}
    """

    def __init__(self, cfg: Any) -> None:
        try:
            from procgen import ProcgenEnv  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "procgen is not installed. ProcgenMazeEnv requires Python 3.10. "
                "See plan/2026-03-16-contextual-shards.md."
            ) from e

        env_cfg = cfg.environment
        self._env = ProcgenEnv(
            num_envs=1,
            env_name="maze",
            num_levels=env_cfg.num_levels,
            start_level=env_cfg.seed,
            distribution_mode=env_cfg.distribution_mode,
        )
        self._obs: Optional[np.ndarray] = None
        self._info: Dict = {}

    def reset(self) -> np.ndarray:
        obs = self._env.reset()
        self._obs = obs["rgb"][0]
        return self._obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self._env.step([action])
        self._obs = obs["rgb"][0]
        self._info = info[0] if info else {}
        return self._obs, float(reward[0]), bool(done[0]), self._info

    def cheese_pos(self) -> Tuple[int, int]:
        """Return cheese position from environment info."""
        return self._info.get("cheese_pos", (0, 0))

    def agent_pos(self) -> Tuple[int, int]:
        """Return agent position from environment info."""
        return self._info.get("agent_pos", (0, 0))

    def close(self) -> None:
        self._env.close()
