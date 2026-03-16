import numpy as np
from typing import Tuple
from src.data_module.concept_labeler.base_labeler import BaseConceptLabeler
from src.data_module.concept_labeler import register_labeler

@register_labeler("cheese_direction")
class CheeseDirectionLabeler(BaseConceptLabeler):
    """Angle (radians) from agent to cheese."""

    def label(
        self,
        agent_pos: Tuple[int, int],
        cheese_pos: Tuple[int, int],
        maze_grid: np.ndarray,  # noqa: ARG002
    ) -> float:
        dy = cheese_pos[0] - agent_pos[0]
        dx = cheese_pos[1] - agent_pos[1]
        return float(np.arctan2(dy, dx))
