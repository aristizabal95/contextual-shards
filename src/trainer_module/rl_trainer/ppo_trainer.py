"""PPO trainer scaffold for maze agents with configurable cheese placement.

Note: This module requires stable-baselines3 and procgen (Python 3.10).
It serves as the interface definition for Experiment 3 training runs.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingConfig:
    """Configuration for RL agent training."""
    cheese_mode: str = "uniform"
    grid_size: int = 15
    total_timesteps: int = 5_000_000
    seed: int = 42
    output_path: str = "outputs/checkpoints/agent.pt"
    n_envs: int = 64


class PPOTrainer:
    """PPO trainer for maze agents with configurable cheese placement distribution.

    Wraps stable-baselines3 PPO for maze-agent training.
    The cheese placement distribution is controlled via a callback
    that intercepts episode resets and repositions cheese.

    Requirements (not yet installed due to Python 3.11 incompatibility):
        - stable-baselines3
        - procgen==0.10.7 (requires Python 3.10)
        - procgen-tools (from git)

    Usage (once environment is set up with Python 3.10):
        trainer = PPOTrainer(config)
        trainer.train()
    """

    def __init__(self, config: RLTrainingConfig):
        self.config = config
        logger.info(
            f"PPOTrainer configured: mode={config.cheese_mode}, "
            f"timesteps={config.total_timesteps}"
        )

    def train(self) -> None:
        """Train agent. Requires procgen + stable-baselines3."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required for RL training. "
                "Install in a Python 3.10 environment with: pip install stable-baselines3"
            )
        from src.trainer_module.rl_trainer.cheese_distribution import CheesePlacementDistribution
        dist = CheesePlacementDistribution(self.config.cheese_mode, self.config.grid_size)
        logger.info(f"Corner fraction for mode '{self.config.cheese_mode}': "
                    f"{dist.empirical_corner_fraction():.3f}")
        raise NotImplementedError(
            "Full training loop requires procgen environment (Python 3.10). "
            "See plan/2026-03-16-contextual-shards.md Task 10 for setup instructions."
        )
