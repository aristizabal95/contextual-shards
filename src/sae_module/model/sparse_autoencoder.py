"""Sparse Autoencoder following Bricken et al. (2023).

Architecture:
  x -> centered -> W_enc -> ReLU -> features (sparse) -> W_dec -> reconstruction

Training objective:
  L = MSE(x, recon) + l1_coef * mean(|features|)

Key detail: decoder columns are constrained to unit norm after each gradient step.
This prevents the model from trivially minimizing L1 by scaling weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.sae_module import register_sae


@register_sae("standard")
class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with L1 sparsity penalty (Bricken et al., 2023).

    Args:
        d_input: dimensionality of the input activations
        expansion_factor: d_hidden = d_input * expansion_factor (8-16x per paper)
        l1_coef: weight of the L1 sparsity penalty on feature activations
    """

    def __init__(
        self,
        d_input: int = 256,
        expansion_factor: int = 8,
        l1_coef: float = 0.01,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_input * expansion_factor
        self.l1_coef = l1_coef

        # Pre-encoder bias (learned centering)
        self.b_pre = nn.Parameter(torch.zeros(d_input))

        # Encoder: input -> hidden features
        self.W_enc = nn.Linear(d_input, self.d_hidden, bias=True)

        # Decoder: hidden features -> reconstruction (no bias — b_pre handles offset)
        self.W_dec = nn.Linear(self.d_hidden, d_input, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with unit-norm decoder columns."""
        nn.init.kaiming_uniform_(self.W_enc.weight)
        nn.init.zeros_(self.W_enc.bias)
        nn.init.kaiming_uniform_(self.W_dec.weight)
        self.normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode centered input to sparse feature activations.

        Args:
            x: (B, d_input) input activations

        Returns:
            features: (B, d_hidden) non-negative sparse features
        """
        return F.relu(self.W_enc(x - self.b_pre))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space.

        Args:
            features: (B, d_hidden) sparse activations

        Returns:
            reconstruction: (B, d_input)
        """
        return self.W_dec(features) + self.b_pre

    def forward(self, x: torch.Tensor):
        """Full forward pass.

        Returns:
            Tuple[reconstruction (B, d_input), features (B, d_hidden)]
        """
        features = self.encode(x)
        recon = self.decode(features)
        return recon, features

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss: MSE reconstruction + L1 sparsity.

        Args:
            x: original input (B, d_input)
            recon: reconstructed input (B, d_input)
            features: sparse feature activations (B, d_hidden)

        Returns:
            scalar loss tensor
        """
        mse = F.mse_loss(recon, x)
        l1 = self.l1_coef * features.abs().mean()
        return mse + l1

    def normalize_decoder(self) -> None:
        """Constrain decoder columns to unit norm.

        Must be called after each gradient step to prevent trivial L1 minimization.
        """
        with torch.no_grad():
            self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=0)

    def feature_sparsity(self, x: torch.Tensor) -> float:
        """Fraction of feature activations that are zero (higher = sparser)."""
        with torch.no_grad():
            _, features = self.forward(x)
        return float((features == 0).float().mean().item())
