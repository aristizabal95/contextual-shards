"""Train Sparse Autoencoder on collected activations (Experiment 4).

Usage (after collect_activations.py):
    uv run python run/pipeline/training/train_sae.py
    uv run python run/pipeline/training/train_sae.py \
        --layer embedder.fc \
        --activations data/processed/activations_impala.h5 \
        --n_epochs 20 \
        --l1_coef 0.01

The SAE is trained on activations from the layer identified in Experiment 1.
"""
import argparse
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAE on IMPALA layer activations.")
    parser.add_argument("--layer", default="embedder.fc",
                        help="Layer name whose activations to train on")
    parser.add_argument("--activations", default="data/processed/activations_impala.h5")
    parser.add_argument("--output", default="outputs/checkpoints/sae.pt")
    parser.add_argument("--expansion_factor", type=int, default=8,
                        help="SAE hidden dim = d_input * expansion_factor")
    parser.add_argument("--l1_coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not os.path.exists(args.activations):
        raise FileNotFoundError(
            f"Activations file not found: {args.activations}\n"
            "Run collect_activations.py first."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset
    from src.sae_module.model.sparse_autoencoder import SparseAutoencoder
    from src.sae_module.training.sae_trainer import SAETrainer

    logger.info(f"Loading activations for layer '{args.layer}' from {args.activations}")
    with HDF5ActivationDataset(args.activations, mode="r") as ds:
        acts = ds.get_all_activations(args.layer)

    X = torch.from_numpy(acts).float()
    X = X.reshape(len(X), -1)  # flatten spatial dims if any
    d_input = X.shape[1]
    logger.info(f"Activation tensor: {X.shape} (d_input={d_input})")

    sae = SparseAutoencoder(
        d_input=d_input,
        expansion_factor=args.expansion_factor,
        l1_coef=args.l1_coef,
    )
    trainer = SAETrainer(
        sae=sae,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )

    logger.info(f"Training SAE: d_input={d_input}, "
                f"d_hidden={d_input * args.expansion_factor}, "
                f"l1={args.l1_coef}, epochs={args.n_epochs}")
    losses = trainer.train_on_tensor(X)
    logger.info(f"Final loss: {losses[-1]:.6f}")

    trainer.save(args.output)
    print(f"SAE saved to {args.output}")


if __name__ == "__main__":
    main()
