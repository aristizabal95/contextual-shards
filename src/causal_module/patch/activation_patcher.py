from contextlib import contextmanager
from typing import Callable
import torch
import torch.nn as nn


class ActivationPatcher:
    """Applies activation patches during forward passes via PyTorch hooks.

    Supports:
    - project_out: remove a direction from activations (shard suppression)
    - project_add: amplify a direction in activations
    - make_zero_patch: zero out layer output
    - make_restore_patch: restore to a saved clean activation
    - make_suppress_patch: project out a shard direction
    - patch_layer: context manager to apply any patch_fn to a named layer
    """

    def __init__(self, model: nn.Module):
        self._model = model

    def project_out(self, x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Project out a direction vector from activations (shard suppression).

        Works on batched tensors of any shape by flattening to (B, D) and reshaping back.
        """
        orig_shape = x.shape
        flat = x.reshape(x.shape[0], -1).float()
        d = direction.reshape(-1).float()
        d = d / (d.norm() + 1e-8)
        # Gram-Schmidt projection: remove component along d
        proj_coeff = flat @ d  # (B,)
        proj = proj_coeff.unsqueeze(1) * d.unsqueeze(0)  # (B, D)
        return (flat - proj).reshape(orig_shape)

    def project_add(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Add a scaled direction vector to activations (shard amplification)."""
        orig_shape = x.shape
        flat = x.reshape(x.shape[0], -1).float()
        d = direction.reshape(-1).float()
        d = d / (d.norm() + 1e-8)
        return (flat + scale * d.unsqueeze(0)).reshape(orig_shape)

    def make_zero_patch(self) -> Callable:
        """Replace layer output with zeros."""
        return lambda x: torch.zeros_like(x)

    def make_restore_patch(self, clean_activation: torch.Tensor) -> Callable:
        """Restore layer output to a saved clean activation (for causal tracing)."""
        saved = clean_activation.detach().clone()

        def patch(x: torch.Tensor) -> torch.Tensor:
            # Handle batch size mismatch gracefully
            if x.shape == saved.shape:
                return saved
            # If batch sizes differ, expand/repeat as needed
            if saved.shape[0] == 1 and x.shape[0] > 1:
                return saved.expand_as(x)
            return saved

        return patch

    def make_suppress_patch(self, direction: torch.Tensor) -> Callable:
        """Project out a shard direction from activations."""
        direction = direction.detach()

        def patch(x: torch.Tensor) -> torch.Tensor:
            return self.project_out(x, direction)

        return patch

    @contextmanager
    def patch_layer(self, layer_name: str, patch_fn: Callable):
        """Context manager to apply patch_fn to a named layer's output.

        Args:
            layer_name: key in model.named_modules(); empty string "" = model itself
            patch_fn: function (tensor) -> tensor applied to the layer output
        """
        named = dict(self._model.named_modules())
        if layer_name == "" or layer_name not in named:
            target = self._model
        else:
            target = named[layer_name]

        hook = target.register_forward_hook(
            lambda module, inp, out: patch_fn(out)
        )
        try:
            yield
        finally:
            hook.remove()
