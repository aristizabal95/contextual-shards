import numpy as np
import pytest
from src.probe_module import ProbeFactory
from src.probe_module.evaluator.probe_evaluator import ProbeEvaluator
from src.probe_module.probe.mdl_probe import MDLProbe


def _make_separable_data(n: int = 200, d: int = 16):
    """Two linearly separable classes."""
    np.random.seed(42)
    X = np.random.randn(n, d).astype(np.float32)
    y = (X[:, 0] > 0).astype(float)
    return X, y


def test_linear_probe_factory():
    probe = ProbeFactory("linear")
    assert probe is not None


def test_linear_probe_fits_and_scores():
    probe = ProbeFactory("linear")()
    X, y = _make_separable_data()
    probe.fit(X[:150], y[:150])
    acc = probe.score(X[150:], y[150:])
    assert acc > 0.85, f"Expected >0.85 on separable data, got {acc:.3f}"


def test_linear_probe_predict_shape():
    probe = ProbeFactory("linear")()
    X, y = _make_separable_data()
    probe.fit(X[:150], y[:150])
    preds = probe.predict(X[150:])
    assert preds.shape == y[150:].shape


def test_linear_probe_handles_2d_activations():
    """Probe should flatten spatial activations automatically."""
    probe = ProbeFactory("linear")()
    X = np.random.randn(100, 16, 8, 8).astype(np.float32)
    y = (np.random.rand(100) > 0.5).astype(float)
    probe.fit(X[:80], y[:80])
    acc = probe.score(X[80:], y[80:])
    assert 0.0 <= acc <= 1.0


def test_mdl_probe_factory():
    probe = ProbeFactory("mdl")
    assert probe is not None


def test_mdl_probe_fits_and_has_codelength():
    probe = MDLProbe()
    X, y = _make_separable_data()
    probe.fit(X, y)
    assert probe.codelength < float("inf")
    assert probe.codelength > 0


def test_probe_evaluator_context_split():
    np.random.seed(0)
    evaluator = ProbeEvaluator(probe_name="linear", n_trials=3)
    X = np.random.randn(200, 8).astype(np.float32)
    # Make in-context linearly separable; out-of-context labels are random binary (no signal)
    y = np.zeros(200)
    y[:100] = (X[:100, 0] > 0).astype(float)
    y[100:] = np.random.randint(0, 2, size=100).astype(float)  # out-of-context: no signal
    context_mask = np.array([True] * 100 + [False] * 100)
    results = evaluator.evaluate_context_split(X, y, context_mask, n_trials=3)
    assert "in_context_acc" in results
    assert "out_context_acc" in results
    assert 0.0 <= results["in_context_acc"] <= 1.0


def test_probe_evaluator_all_layers():
    np.random.seed(1)
    evaluator = ProbeEvaluator(probe_name="linear", n_trials=2)
    acts = {
        "block1": np.random.randn(100, 16).astype(np.float32),
        "fc": np.random.randn(100, 256).astype(np.float32),
    }
    y = (np.random.rand(100) > 0.5).astype(float)
    context_mask = np.ones(100, dtype=bool)
    context_mask[50:] = False
    results = evaluator.evaluate_all_layers(acts, y, context_mask)
    assert "block1" in results
    assert "fc" in results


def test_bonferroni_threshold():
    evaluator = ProbeEvaluator()
    threshold = evaluator.bonferroni_threshold(n_comparisons=10, alpha=0.01)
    assert threshold == pytest.approx(0.001)


def test_probe_factory_raises_unknown():
    with pytest.raises(ValueError, match="Unknown probe"):
        ProbeFactory("nonexistent_probe")
