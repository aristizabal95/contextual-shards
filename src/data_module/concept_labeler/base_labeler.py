from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseConceptLabeler(ABC):
    """Computes scalar/categorical concept label from environment state."""

    @abstractmethod
    def label(
        self,
        agent_pos: Tuple[int, int],
        cheese_pos: Tuple[int, int],
        maze_grid: np.ndarray,
    ) -> float:
        """Return continuous or binary label for the concept."""
        ...
