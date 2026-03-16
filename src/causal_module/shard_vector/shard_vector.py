import numpy as np
import torch
from typing import Dict, Union


class ShardVector:
    """Computes shard direction vectors as mean activation differences.

    Following Turner et al. (2023): shard vector = mean(activations | concept ON)
    minus mean(activations | concept OFF). This direction vector represents
    the concept's encoding direction in the layer's activation space.
    """

    def compute(
        self,
        context_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        baseline_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        """Compute mean activation difference vectors per layer.

        Args:
            context_activations: {layer: (N, ...)} when concept IS present
            baseline_activations: {layer: (N, ...)} when concept is ABSENT

        Returns:
            {layer: mean_diff_vector} — same shape as a single activation
        """
        vectors: Dict[str, torch.Tensor] = {}
        for layer in context_activations:
            ctx = context_activations[layer]
            base = baseline_activations[layer]
            if isinstance(ctx, np.ndarray):
                ctx = torch.from_numpy(ctx)
            if isinstance(base, np.ndarray):
                base = torch.from_numpy(base)
            vectors[layer] = ctx.float().mean(0) - base.float().mean(0)
        return vectors

    def normalize(self, vectors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return unit-norm versions of shard vectors."""
        return {
            layer: v / (v.norm() + 1e-8)
            for layer, v in vectors.items()
        }
