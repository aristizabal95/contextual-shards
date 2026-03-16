"""SAE training loop for Experiment 4."""
import logging
from typing import Optional
import torch
from src.sae_module.model.sparse_autoencoder import SparseAutoencoder

logger = logging.getLogger(__name__)


class SAETrainer:
    """Trains a SparseAutoencoder on stored layer activations.

    Reads activations from an HDF5ActivationDataset (or any array-like),
    runs Adam optimization with L1+MSE loss, normalizes decoder columns
    after each step.

    Args:
        sae: initialized SparseAutoencoder model
        lr: learning rate (default 1e-4 per Bricken et al.)
        n_epochs: training epochs
        batch_size: mini-batch size
        device: torch device string
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        lr: float = 1e-4,
        n_epochs: int = 10,
        batch_size: int = 512,
        device: Optional[str] = None,
    ):
        self.sae = sae
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sae.to(self.device)

    def train_on_tensor(self, X: torch.Tensor) -> list:
        """Train SAE on a pre-loaded tensor of activations.

        Args:
            X: (N, d_input) tensor of activations

        Returns:
            list of per-epoch mean losses
        """
        X = X.to(self.device).float()
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.lr)
        epoch_losses = []

        for epoch in range(self.n_epochs):
            perm = torch.randperm(len(X))
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(X), self.batch_size):
                batch = X[perm[start : start + self.batch_size]]
                optimizer.zero_grad()
                recon, features = self.sae(batch)
                loss = self.sae.loss(batch, recon, features)
                loss.backward()
                optimizer.step()
                self.sae.normalize_decoder()
                total_loss += loss.item()
                n_batches += 1

            mean_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(mean_loss)
            logger.info(f"Epoch {epoch + 1}/{self.n_epochs}: loss={mean_loss:.6f}")

        return epoch_losses

    def save(self, path: str) -> None:
        """Save SAE state dict to path."""
        torch.save(self.sae.state_dict(), path)
        logger.info(f"Saved SAE to {path}")

    def load(self, path: str) -> None:
        """Load SAE state dict from path."""
        state = torch.load(path, map_location=self.device)
        self.sae.load_state_dict(state)
        logger.info(f"Loaded SAE from {path}")
