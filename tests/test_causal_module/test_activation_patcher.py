import torch
import pytest
import numpy as np
from src.causal_module.patch.activation_patcher import ActivationPatcher


def _simple_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())


def test_project_out_removes_component():
    model = torch.nn.Identity()
    patcher = ActivationPatcher(model)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x = torch.tensor([[3.0, 1.0, 1.0, 1.0], [2.0, 2.0, 0.0, 1.0]])
    result = patcher.project_out(x, direction)
    # Component along direction (dim 0) should be ~0
    assert result[:, 0].abs().max().item() < 1e-5
    # Other components should be unchanged
    assert torch.allclose(result[:, 1:], x[:, 1:])


def test_project_add_amplifies_component():
    model = torch.nn.Identity()
    patcher = ActivationPatcher(model)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    result = patcher.project_add(x, direction, scale=5.0)
    assert result[0, 0].item() == pytest.approx(5.0, abs=1e-4)


def test_make_zero_patch():
    model = _simple_model()
    patcher = ActivationPatcher(model)
    zero_patch = patcher.make_zero_patch()
    x = torch.ones(2, 4)
    with patcher.patch_layer("0", zero_patch):
        out = model(x)
    # After zeroing linear output, ReLU(0) = 0
    assert torch.allclose(out, torch.zeros_like(out))


def test_make_restore_patch():
    model = _simple_model()
    patcher = ActivationPatcher(model)
    saved = torch.ones(2, 8) * 99.0
    restore_patch = patcher.make_restore_patch(saved)
    x = torch.zeros(2, 4)
    with patcher.patch_layer("0", restore_patch):
        out = model(x)
    # After restoring saved (all 99s), ReLU(99) = 99
    assert torch.allclose(out, torch.ones_like(out) * 99.0)


def test_make_suppress_patch():
    model = torch.nn.Identity()
    patcher = ActivationPatcher(model)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    suppress_patch = patcher.make_suppress_patch(direction)
    x = torch.tensor([[5.0, 3.0, 2.0, 1.0]])
    with patcher.patch_layer("", suppress_patch):
        out = model(x)
    assert out[0, 0].abs().item() < 1e-5


def test_patch_layer_context_manager_removes_hook():
    """Hook should be removed after exiting context manager."""
    model = _simple_model()
    patcher = ActivationPatcher(model)
    zero_patch = patcher.make_zero_patch()
    x = torch.ones(1, 4)
    with patcher.patch_layer("0", zero_patch):
        out_patched = model(x)
    out_normal = model(x)
    # Patched output should differ from normal (zero vs non-zero)
    assert not torch.allclose(out_patched, out_normal)
