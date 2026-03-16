import numpy as np
from typing import Tuple
from src.data_module.concept_labeler.base_labeler import BaseConceptLabeler
from src.data_module.concept_labeler import register_labeler

@register_labeler("cheese_presence")
class CheesePresenceLabeler(BaseConceptLabeler):
    """Binary: 1 if cheese within proximity_threshold steps, else 0."""

    def __init__(self, threshold: int = 5):
        self.threshold = threshold

    def label(
        self,
        agent_pos: Tuple[int, int],
        cheese_pos: Tuple[int, int],
        maze_grid: np.ndarray,
    ) -> float:
        dist = np.sqrt((agent_pos[0] - cheese_pos[0])**2 + (agent_pos[1] - cheese_pos[1])**2)
        return 1.0 if dist <= self.threshold else 0.0
