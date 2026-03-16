"""Train Sparse Autoencoder on collected activations (Experiment 4).

Usage (after collect_activations.py):
    uv run python run/pipeline/training/train_sae.py

The SAE is trained on activations from the layer identified in Experiment 1.
"""
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    print("Experiment 4: SAE Training")
    print("=" * 50)
    print("Prerequisites:")
    print("  - collect_activations.py completed (>100k samples in data/processed/)")
    print("  - Experiment 1 shard layer identified (e.g., 'fc')")
    print()
    print("Quick start:")
    print("  from src.sae_module.model.sparse_autoencoder import SparseAutoencoder")
    print("  from src.sae_module.training.sae_trainer import SAETrainer")
    print("  from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset")
    print()
    print("  sae = SparseAutoencoder(d_input=256, expansion_factor=8, l1_coef=0.01)")
    print("  trainer = SAETrainer(sae, lr=1e-4, n_epochs=10)")
    print("  ds = HDF5ActivationDataset('data/processed/activations.h5', mode='r')")
    print("  X = torch.from_numpy(ds.get_all_activations('fc'))")
    print("  trainer.train_on_tensor(X)")
    print("  trainer.save('outputs/checkpoints/sae.pt')")


if __name__ == "__main__":
    main()
