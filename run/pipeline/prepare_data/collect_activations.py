"""Collect neural network activations from a trained agent in the maze environment.

Saves activations + concept labels to an HDF5 file for use in probing and SAE training.

Usage:
    uv run python run/pipeline/prepare_data/collect_activations.py

Prerequisites:
    - Agent checkpoint at data/raw/maze_policy.pt
    - procgen + procgen-tools installed (Python 3.10 environment)
"""
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    print("Activation Collection Pipeline")
    print("=" * 50)
    print("Prerequisites:")
    print("  - data/raw/maze_policy.pt (download from procgen-tools repo)")
    print("  - procgen==0.10.7 (Python 3.10 environment)")
    print("  - procgen-tools (pip install git+https://github.com/aristizabal95/procgen-tools)")
    print()
    print("Quick start (Python 3.10 env):")
    print("  from src.environment_module.maze.maze_env import ProcgenMazeEnv")
    print("  from src.agent_module.policy.impala_agent import ImpalaAgent")
    print("  from src.data_module.concept_labeler import LabelerFactory")
    print("  from src.data_module.rollout_collector.rollout_collector import RolloutCollector")
    print("  from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset")
    print()
    print("  env = ProcgenMazeEnv(cfg)")
    print("  agent = ImpalaAgent(cfg)")
    print("  labelers = {name: LabelerFactory(name)() for name in ['cheese_presence', 'corner_proximity']}")
    print("  collector = RolloutCollector(env, agent, labelers, n_steps=100000)")
    print("  batch = collector.collect()")
    print("  with HDF5ActivationDataset('data/processed/activations_impala.h5', 'w') as ds:")
    print("      ds.write_batch(batch.activations, batch.labels)")


if __name__ == "__main__":
    main()
