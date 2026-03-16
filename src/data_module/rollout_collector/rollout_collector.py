"""RolloutCollector — drives an agent through an environment and records data.

Collects per-step observations, activations (via ActivationRecorder), and
concept labels (via LabelerFactory), saving everything to an HDF5 dataset.
"""
import logging
from typing import Any, Dict, List

import numpy as np

from src.agent_module.hooks.activation_hooks import ActivationRecorder
from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset
from src.data_module.concept_labeler import LabelerFactory
from src.environment_module.base_env import BaseEnv

logger = logging.getLogger(__name__)


class RolloutCollector:
    """Collects rollouts from an agent + environment, recording activations and labels.

    Args:
        agent: Loaded BaseAgent with a PyTorch model (agent.model must not be None).
        env: BaseEnv instance.
        layer_names: Names of layers to record activations from.
        concept_names: Names of concept labelers to apply (e.g. "cheese_presence").
        batch_size: Number of steps to accumulate before writing to HDF5.
    """

    def __init__(
        self,
        agent: Any,
        env: BaseEnv,
        layer_names: List[str],
        concept_names: List[str],
        batch_size: int = 256,
    ) -> None:
        self._agent = agent
        self._env = env
        self._layer_names = layer_names
        self._labelers = {name: LabelerFactory(name)() for name in concept_names}
        self._batch_size = batch_size
        self._recorder = ActivationRecorder(agent.model, layer_names)

    def collect(
        self,
        output_path: str,
        n_steps: int = 10000,
        max_episode_steps: int = 256,
    ) -> None:
        """Run the agent for n_steps and write data to HDF5.

        Args:
            output_path: Path to write the HDF5 file.
            n_steps: Total number of environment steps to collect.
            max_episode_steps: Max steps per episode before forced reset.
        """
        with HDF5ActivationDataset(output_path, mode="w") as dataset:
            act_buffer: Dict[str, List[np.ndarray]] = {n: [] for n in self._layer_names}
            label_buffer: Dict[str, List[float]] = {n: [] for n in self._labelers}

            obs = self._env.reset()
            episode_steps = 0
            steps_collected = 0

            while steps_collected < n_steps:
                with self._recorder.record() as activations:
                    action = self._agent.act(obs)

                # Get current positions for concept labeling
                agent_pos = self._env.agent_pos()
                cheese_pos = self._env.cheese_pos()

                # Record activations
                for layer in self._layer_names:
                    if layer in activations:
                        act_buffer[layer].append(
                            activations[layer].squeeze(0).numpy()
                        )

                # Compute concept labels
                for name, labeler in self._labelers.items():
                    label = labeler.label(
                        agent_pos=agent_pos,
                        cheese_pos=cheese_pos,
                        maze_grid=np.zeros((15, 15), dtype=np.int64),  # placeholder
                    )
                    label_buffer[name].append(label)

                obs, _reward, done, _info = self._env.step(action)
                episode_steps += 1
                steps_collected += 1

                if done or episode_steps >= max_episode_steps:
                    obs = self._env.reset()
                    episode_steps = 0

                if steps_collected % self._batch_size == 0:
                    self._flush(dataset, act_buffer, label_buffer)
                    logger.info(f"Collected {steps_collected}/{n_steps} steps")

            # Flush remaining
            if any(len(v) > 0 for v in act_buffer.values()):
                self._flush(dataset, act_buffer, label_buffer)

        logger.info(f"Saved {steps_collected} steps to {output_path}")

    def _flush(
        self,
        dataset: HDF5ActivationDataset,
        act_buffer: Dict[str, List[np.ndarray]],
        label_buffer: Dict[str, List[float]],
    ) -> None:
        """Write buffered data to HDF5 and clear buffers."""
        if not any(len(v) > 0 for v in act_buffer.values()):
            return

        acts = {k: np.stack(v) for k, v in act_buffer.items() if v}
        labels = {k: np.array(v) for k, v in label_buffer.items() if v}

        dataset.write_batch(acts, labels)

        for v in act_buffer.values():
            v.clear()
        for v in label_buffer.values():
            v.clear()
