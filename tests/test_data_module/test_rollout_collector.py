"""Tests for RolloutCollector using mock agent and environment."""
import os
import tempfile
from typing import Dict, Tuple

import numpy as np
import torch.nn as nn

from src.data_module.rollout_collector.rollout_collector import RolloutCollector
from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset
from src.data_module.concept_labeler import register_labeler
from src.data_module.concept_labeler.base_labeler import BaseConceptLabeler
from src.environment_module.base_env import BaseEnv


# ── Mock objects ────────────────────────────────────────────────────────────


@register_labeler("_test_mock_concept")
class MockLabeler(BaseConceptLabeler):
    def label(self, agent_pos, cheese_pos, maze_grid) -> float:
        return 1.0


class MockEnv(BaseEnv):
    def reset(self) -> np.ndarray:
        return np.zeros((3, 8, 8), dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        return np.zeros((3, 8, 8), dtype=np.float32), 0.0, False, {}

    def cheese_pos(self) -> Tuple[int, int]:
        return (2, 3)

    def agent_pos(self) -> Tuple[int, int]:
        return (1, 1)

    def close(self) -> None:
        pass


class MockModel(nn.Module):
    """Tiny model with a named layer for hook testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 4)

    def forward(self, x):
        return self.fc(x.reshape(x.shape[0], -1))


class MockAgent:
    """Agent that exposes a MockModel and runs it on act() so hooks fire."""
    def __init__(self):
        self.model = MockModel()

    def act(self, obs: np.ndarray) -> int:
        import torch
        x = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        return int(out.argmax().item())


# ── Tests ───────────────────────────────────────────────────────────────────


def test_rollout_collector_creates_hdf5():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.h5")
        collector = RolloutCollector(
            agent=MockAgent(),
            env=MockEnv(),
            layer_names=["fc"],
            concept_names=["_test_mock_concept"],
            batch_size=10,
        )
        collector.collect(path, n_steps=20, max_episode_steps=50)
        assert os.path.exists(path)


def test_rollout_collector_step_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.h5")
        n_steps = 30
        collector = RolloutCollector(
            agent=MockAgent(),
            env=MockEnv(),
            layer_names=["fc"],
            concept_names=["_test_mock_concept"],
            batch_size=16,
        )
        collector.collect(path, n_steps=n_steps)
        with HDF5ActivationDataset(path, mode="r") as ds:
            assert len(ds) == n_steps


def test_rollout_collector_label_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.h5")
        collector = RolloutCollector(
            agent=MockAgent(),
            env=MockEnv(),
            layer_names=["fc"],
            concept_names=["_test_mock_concept"],
            batch_size=8,
        )
        collector.collect(path, n_steps=16)
        with HDF5ActivationDataset(path, mode="r") as ds:
            labels = ds.get_all_labels("_test_mock_concept")
        assert labels.shape == (16,)
        assert np.all(labels == 1.0)


def test_rollout_collector_activation_shape():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.h5")
        collector = RolloutCollector(
            agent=MockAgent(),
            env=MockEnv(),
            layer_names=["fc"],
            concept_names=["_test_mock_concept"],
            batch_size=8,
        )
        collector.collect(path, n_steps=16)
        with HDF5ActivationDataset(path, mode="r") as ds:
            acts = ds.get_all_activations("fc")
        assert acts.shape[0] == 16  # 16 steps
        assert acts.ndim == 2  # (steps, features)
