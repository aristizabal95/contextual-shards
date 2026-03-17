"""Regression tests for CausalTracer.

Covers two historical bugs:

Bug A — "same obs twice"
    Original experiment1_causal.py called:
        tracer.trace(obs_clean, obs_clean, ...)
    Because clean_prob == corrupted_prob, the gap ≈ 1e-8 and every effect is 0.0.

Bug B — "extra unsqueeze"
    ActivationRecorder stores layer outputs WITH the batch dimension (shape (1, D) for a
    single observation). The tracer previously called:
        make_restore_patch(clean_act.unsqueeze(0))   # ← adds a redundant outer dim
    This produced a saved tensor of shape (1, 1, D), whose shape never matched the hook
    input (1, D), so the fallback path returned the wrong-shaped tensor and the patch
    had no effect on model output (all causal effects remained 0.0).
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from src.agent_module.hooks.activation_hooks import ActivationRecorder
from src.causal_module.tracing.causal_tracer import CausalTracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockAgent:
    """Minimal AgentProtocol implementation backed by a deterministic MLP."""

    def __init__(self, model: nn.Module, layer_names: List[str]) -> None:
        self._model = model
        self._layer_names = layer_names

    def act_with_activations(self, obs: np.ndarray) -> Tuple[int, Dict[str, torch.Tensor]]:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        recorder = ActivationRecorder(self._model, self._layer_names)
        with recorder.record() as acts:
            with torch.no_grad():
                logits = self._model(obs_t)
        action = int(logits.argmax(dim=-1).item())
        return action, dict(acts)

    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            logits = self._model(obs_t)
        return torch.softmax(logits, dim=-1)[0].numpy()

    @property
    def policy(self) -> nn.Module:
        return self._model


def _build_agent(input_dim: int = 8, hidden: int = 16, n_actions: int = 5):
    """Two-layer MLP: Linear → ReLU → Linear with fixed random weights."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden, bias=False),
        nn.ReLU(),
        nn.Linear(hidden, n_actions, bias=False),
    )
    with torch.no_grad():
        for p in model.parameters():
            nn.init.normal_(p, std=1.0)
    model.eval()
    layer_names = ["0", "2"]  # first and last Linear
    return model, _MockAgent(model, layer_names), layer_names


# ---------------------------------------------------------------------------
# Bug A regression — identical observations must yield all-zero effects
# ---------------------------------------------------------------------------

def test_trace_same_obs_produces_zero_effects():
    """Regression for Bug A: passing obs_clean as both args collapses gap to ~0.

    When the same observation is used for both the clean and corrupted run:
      clean_prob == corrupted_prob  →  gap = 1e-8
      restored_prob == corrupted_prob (patch restores identical activation)
      effect = (restored_prob - corrupted_prob) / gap = 0.0

    This test documents that behaviour so a re-introduction of the same-obs
    pattern is immediately detected.
    """
    _, agent, layer_names = _build_agent()
    tracer = CausalTracer(agent, layer_names)

    rng = np.random.default_rng(0)
    obs = rng.standard_normal(8).astype(np.float32)

    effects = tracer.trace(obs, obs, target_action_idx=0)

    for layer, eff in effects.items():
        assert eff == pytest.approx(0.0, abs=1e-6), (
            f"Layer '{layer}': expected 0.0 when clean==corrupted (Bug A), got {eff}"
        )


# ---------------------------------------------------------------------------
# Bug B regressions — activation recorder shape and patch correctness
# ---------------------------------------------------------------------------

def test_activation_recorder_stores_batch_dimension():
    """ActivationRecorder stores the raw hook output, which includes the batch dim.

    With batch=1 the shape is (1, D) for a Linear layer.  CausalTracer must NOT
    call .unsqueeze(0) on this tensor before passing it to make_restore_patch —
    the batch dimension is already present.
    """
    model, _, layer_names = _build_agent()
    obs_t = torch.randn(1, 8)  # explicit batch dim = 1

    recorder = ActivationRecorder(model, layer_names)
    with recorder.record() as acts:
        with torch.no_grad():
            model(obs_t)

    for name in layer_names:
        act = acts[name]
        assert act.dim() >= 2, (
            f"Layer '{name}' activation is {act.dim()}D — expected batch dim to be present."
        )
        assert act.shape[0] == 1, (
            f"Layer '{name}' batch dim should be 1, got shape {tuple(act.shape)}. "
            "Do NOT call .unsqueeze(0) before make_restore_patch."
        )


def test_trace_different_obs_yields_nonzero_effect():
    """Regression for Bug B: when obs differ, the restore patch must change model output.

    If clean_act.unsqueeze(0) is used (wrong), the saved tensor has shape (1, 1, D)
    while the hook input has shape (1, D).  Neither shape-match branch in
    make_restore_patch fires, so the function returns the wrong-shaped tensor;
    the downstream forward pass uses the original (corrupted) activations and
    causal effects stay at 0.0.

    With the fix (no extra unsqueeze), shapes match, the patch is applied, and at
    least one layer must show a non-negligible causal effect.
    """
    _, agent, layer_names = _build_agent()
    tracer = CausalTracer(agent, layer_names)

    rng = np.random.default_rng(1)
    obs_clean = rng.standard_normal(8).astype(np.float32)
    obs_corrupted = rng.standard_normal(8).astype(np.float32)

    action_clean = int(agent.get_action_probs(obs_clean).argmax())
    effects = tracer.trace(obs_clean, obs_corrupted, target_action_idx=action_clean)

    assert any(abs(v) > 1e-4 for v in effects.values()), (
        "All causal effects are ~0.0 for distinct observations. "
        "The restore patch is likely not being applied — verify that "
        "CausalTracer does NOT call clean_act.unsqueeze(0) (Bug B)."
    )


def test_trace_first_layer_gives_full_recovery():
    """Patching the first layer with clean activations should recover ~clean_prob.

    For a feedforward network, restoring the output of layer 0 means the entire
    downstream computation operates on clean features, so:
        restored_prob ≈ clean_prob  →  causal_effect ≈ 1.0
    """
    _, agent, layer_names = _build_agent()
    tracer = CausalTracer(agent, layer_names)

    # Use negated obs to guarantee very different activation patterns
    rng = np.random.default_rng(2)
    obs_clean = rng.standard_normal(8).astype(np.float32)
    obs_corrupted = -obs_clean

    action_clean = int(agent.get_action_probs(obs_clean).argmax())
    clean_prob = float(agent.get_action_probs(obs_clean)[action_clean])
    corrupted_prob = float(agent.get_action_probs(obs_corrupted)[action_clean])

    if abs(clean_prob - corrupted_prob) < 0.02:
        pytest.skip("clean/corrupted probs too similar for this test to be meaningful")

    effects = tracer.trace(obs_clean, obs_corrupted, target_action_idx=action_clean)

    first_layer = layer_names[0]
    assert effects[first_layer] == pytest.approx(1.0, abs=0.15), (
        f"Expected ~1.0 recovery at first layer '{first_layer}', "
        f"got {effects[first_layer]:.4f}. "
        "Full recovery is expected because all downstream computation "
        "receives clean activations when the first layer is patched."
    )
