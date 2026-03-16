from abc import ABC, abstractmethod
import numpy as np


class BaseProbe(ABC):
    """Abstract base class for concept probes."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the probe on training data."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        ...

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        ...
