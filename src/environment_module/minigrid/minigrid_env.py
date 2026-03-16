"""MiniGridEnv — wraps gymnasium + minigrid for Exp 5 generalization."""
from typing import Any, Dict, Tuple

import numpy as np

from src.environment_module import register_env
from src.environment_module.base_env import BaseEnv


@register_env("minigrid")
class MiniGridEnv(BaseEnv):
    """Wraps a MiniGrid gymnasium environment.

    Args:
        cfg: Config with cfg.environment.{env_id, seed}
    """

    def __init__(self, cfg: Any) -> None:
        try:
            import gymnasium as gym  # type: ignore[import]
            import minigrid as _minigrid  # noqa: F401  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "gymnasium and minigrid must be installed for MiniGridEnv."
            ) from e

        env_cfg = cfg.environment
        self._env = gym.make(getattr(env_cfg, "env_id", "MiniGrid-Empty-5x5-v0"))
        self._cheese_pos: Tuple[int, int] = (0, 0)
        self._agent_pos: Tuple[int, int] = (0, 0)

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        return self._extract_obs(obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return self._extract_obs(obs), float(reward), bool(done), info

    def _extract_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            return obs.get("image", np.zeros((7, 7, 3), dtype=np.uint8))
        return np.asarray(obs)

    def cheese_pos(self) -> Tuple[int, int]:
        return self._cheese_pos

    def agent_pos(self) -> Tuple[int, int]:
        return self._agent_pos

    def close(self) -> None:
        self._env.close()
