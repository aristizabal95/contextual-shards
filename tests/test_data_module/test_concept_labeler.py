import numpy as np
import pytest
from src.data_module.concept_labeler import LabelerFactory


def test_cheese_presence_true():
    labeler = LabelerFactory("cheese_presence")()
    label = labeler.label(agent_pos=(5, 5), cheese_pos=(5, 6), maze_grid=np.zeros((15, 15)))
    assert label == 1.0


def test_cheese_presence_false():
    labeler = LabelerFactory("cheese_presence")()
    label = labeler.label(agent_pos=(1, 1), cheese_pos=(13, 13), maze_grid=np.zeros((15, 15)))
    assert label == 0.0


def test_cheese_proximity_distance():
    labeler = LabelerFactory("cheese_proximity")()
    d = labeler.label(agent_pos=(5, 5), cheese_pos=(5, 10), maze_grid=np.zeros((15, 15)))
    assert d == pytest.approx(5.0)


def test_cheese_direction_angle():
    labeler = LabelerFactory("cheese_direction")()
    angle = labeler.label(agent_pos=(5, 5), cheese_pos=(5, 6), maze_grid=np.zeros((15, 15)))
    assert -np.pi <= angle <= np.pi


def test_cheese_direction_right():
    labeler = LabelerFactory("cheese_direction")()
    angle = labeler.label(agent_pos=(5, 5), cheese_pos=(5, 6), maze_grid=np.zeros((15, 15)))
    assert angle == pytest.approx(0.0)


def test_corner_proximity_near_corner():
    labeler = LabelerFactory("corner_proximity")()
    d = labeler.label(agent_pos=(1, 13), cheese_pos=(0, 0), maze_grid=np.zeros((15, 15)))
    assert d < 3.0


def test_labeler_factory_raises_unknown():
    with pytest.raises(ValueError, match="Unknown labeler"):
        LabelerFactory("nonexistent_concept")
