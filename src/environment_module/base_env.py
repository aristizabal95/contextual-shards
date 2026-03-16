from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class BaseEnv(ABC):
    """Abstract environment interface for the contextual-shards pipeline."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step environment. Returns (obs, reward, done, info)."""

    @abstractmethod
    def cheese_pos(self) -> Tuple[int, int]:
        """Return (row, col) of cheese in current episode."""

    @abstractmethod
    def agent_pos(self) -> Tuple[int, int]:
        """Return (row, col) of agent in current episode."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
