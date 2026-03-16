import numpy as np
from typing import Tuple
from src.data_module.concept_labeler.base_labeler import BaseConceptLabeler
from src.data_module.concept_labeler import register_labeler

@register_labeler("cheese_proximity")
class CheeseProximityLabeler(BaseConceptLabeler):
    """Euclidean distance from agent to cheese."""

    def label(
        self,
        agent_pos: Tuple[int, int],
        cheese_pos: Tuple[int, int],
        maze_grid: np.ndarray,
    ) -> float:
        return float(np.sqrt((agent_pos[0] - cheese_pos[0])**2 + (agent_pos[1] - cheese_pos[1])**2))
