"""Collect neural network activations from a trained IMPALA agent in the maze environment.

Saves activations + concept labels to an HDF5 file for use in probing and SAE training.

Usage:
    uv run python run/pipeline/prepare_data/collect_activations.py
    uv run python run/pipeline/prepare_data/collect_activations.py \
        --checkpoint data/raw/maze_policy.pt \
        --output data/processed/activations_impala.h5 \
        --n_steps 50000
"""
import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

LAYER_NAMES = [
    "embedder.block1", "embedder.block1.res1", "embedder.block1.res2",
    "embedder.block2", "embedder.block2.res1", "embedder.block2.res2",
    "embedder.block3", "embedder.block3.res1", "embedder.block3.res2",
    "embedder.fc",
]

CONCEPT_NAMES = [
    "cheese_presence",
    "cheese_proximity",
    "cheese_direction",
    "corner_proximity",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect activations from IMPALA maze policy.")
    parser.add_argument("--checkpoint", default="data/raw/maze_policy.pt")
    parser.add_argument("--output", default="data/processed/activations_impala.h5")
    parser.add_argument("--n_steps", type=int, default=50000)
    parser.add_argument("--num_levels", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        raise FileNotFoundError(args.checkpoint)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading agent from {args.checkpoint}")
    from src.agent_module.policy.impala_agent import ImpalaAgent

    class Cfg:
        class agent:
            checkpoint_path = args.checkpoint
            layer_names = LAYER_NAMES

    agent = ImpalaAgent(Cfg())

    logger.info("Creating maze environment")
    from src.environment_module.maze.maze_env import ProcgenMazeEnv

    class EnvCfg:
        class environment:
            num_levels = args.num_levels
            seed = args.seed

    env = ProcgenMazeEnv(EnvCfg())

    logger.info(f"Collecting {args.n_steps} steps → {args.output}")
    from src.data_module.rollout_collector.rollout_collector import RolloutCollector

    collector = RolloutCollector(
        agent=agent,
        env=env,
        layer_names=LAYER_NAMES,
        concept_names=CONCEPT_NAMES,
        batch_size=args.batch_size,
    )
    collector.collect(args.output, n_steps=args.n_steps)
    env.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
