"""Train maze agent with configurable cheese placement distribution (Experiment 3).

Usage (Python 3.10 environment with procgen + stable-baselines3):
    uv run python run/pipeline/training/train_rl_agent.py training.cheese_mode=corner_biased
    uv run python run/pipeline/training/train_rl_agent.py training.cheese_mode=uniform
    uv run python run/pipeline/training/train_rl_agent.py training.cheese_mode=anti_corner
"""
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    from src.trainer_module.rl_trainer.cheese_distribution import CheesePlacementDistribution
    from src.trainer_module.rl_trainer.ppo_trainer import PPOTrainer, RLTrainingConfig

    import sys
    mode = "uniform"
    for arg in sys.argv[1:]:
        if arg.startswith("training.cheese_mode="):
            mode = arg.split("=", 1)[1]

    config = RLTrainingConfig(
        cheese_mode=mode,
        output_path=f"outputs/checkpoints/agent_{mode}.pt",
    )
    dist = CheesePlacementDistribution(mode=config.cheese_mode)
    logger.info(f"Training agent with mode={mode}, "
                f"corner_fraction={dist.empirical_corner_fraction():.3f}")
    trainer = PPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
