from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """Abstract agent interface."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        """Select action given observation."""

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint."""
