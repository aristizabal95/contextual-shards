"""Configurable cheese placement distribution for Experiment 3.

Three distributions (per research plan):
  corner_biased: Cheese in top-right 25% of maze 75% of the time; elsewhere 25%.
  uniform: Cheese placed uniformly across all maze positions.
  anti_corner: Cheese in top-right only 10%; bottom-left quadrant 70%; elsewhere 20%.

The maze grid uses row-major indexing where (0,0) is top-left.
Top-right quadrant: row < half, col >= half.
Bottom-left quadrant: row >= half, col < half.
"""
import numpy as np
from typing import Tuple


class CheesePlacementDistribution:
    """Samples cheese (row, col) positions from a configurable maze distribution.

    Args:
        mode: one of "corner_biased", "uniform", "anti_corner"
        grid_size: interior maze size (default 15 for procgen maze)
        seed: optional random seed for reproducibility
    """

    VALID_MODES = ("corner_biased", "uniform", "anti_corner")

    def __init__(self, mode: str, grid_size: int = 15, seed: int = None):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Valid: {self.VALID_MODES}")
        self.mode = mode
        self.grid_size = grid_size
        self.half = grid_size // 2
        self._rng = np.random.default_rng(seed)

    def sample(self) -> Tuple[int, int]:
        """Return (row, col) for a cheese placement in the interior grid.

        Rows and columns are in [0, grid_size - 1].
        Walls are handled by the environment; this sampler returns positions
        in the interior which the environment converts to valid grid cells.
        """
        if self.mode == "corner_biased":
            return self._sample_corner_biased()
        elif self.mode == "uniform":
            return self._sample_uniform()
        else:  # anti_corner
            return self._sample_anti_corner()

    def _sample_uniform(self) -> Tuple[int, int]:
        r = int(self._rng.integers(0, self.grid_size))
        c = int(self._rng.integers(0, self.grid_size))
        return r, c

    def _sample_corner_biased(self) -> Tuple[int, int]:
        """75% top-right quadrant, 25% uniform."""
        if self._rng.random() < 0.75:
            r = int(self._rng.integers(0, self.half))
            c = int(self._rng.integers(self.half, self.grid_size))
            return r, c
        return self._sample_uniform()

    def _sample_anti_corner(self) -> Tuple[int, int]:
        """10% top-right, 70% bottom-left, 20% uniform."""
        p = self._rng.random()
        if p < 0.10:
            r = int(self._rng.integers(0, self.half))
            c = int(self._rng.integers(self.half, self.grid_size))
            return r, c
        elif p < 0.80:
            r = int(self._rng.integers(self.half, self.grid_size))
            c = int(self._rng.integers(0, self.half))
            return r, c
        return self._sample_uniform()

    def empirical_corner_fraction(self, n_samples: int = 10000) -> float:
        """Estimate fraction of samples landing in the top-right quadrant."""
        samples = [self.sample() for _ in range(n_samples)]
        corner_count = sum(1 for r, c in samples if r < self.half and c >= self.half)
        return corner_count / n_samples
