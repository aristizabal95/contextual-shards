import numpy as np
from typing import Tuple
from src.data_module.concept_labeler.base_labeler import BaseConceptLabeler
from src.data_module.concept_labeler import register_labeler

@register_labeler("corner_proximity")
class CornerProximityLabeler(BaseConceptLabeler):
    """Distance from agent to top-right corner of maze."""

    def label(
        self,
        agent_pos: Tuple[int, int],
        cheese_pos: Tuple[int, int],
        maze_grid: np.ndarray,
    ) -> float:
        h, w = maze_grid.shape
        corner = (0, w - 1)
        return float(np.sqrt((agent_pos[0] - corner[0])**2 + (agent_pos[1] - corner[1])**2))
