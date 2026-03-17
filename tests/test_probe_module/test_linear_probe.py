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


def test_circular_context_mask_is_uninformative():
    """Regression for experiment1_probing Bug C: context_mask == binary_labels.

    When context_masks[concept] is set to the same boolean array as binary_labels[concept],
    the 'in-context' test split contains only positive-class samples (y=1) and the
    'out-of-context' split contains only negative-class samples (y=0).

    The result: in_context_acc and out_context_acc are both high and nearly equal —
    they only measure per-class accuracy, NOT contextual encoding in the Shard Theory
    sense.  The gap should be small (< 0.15).
    """
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 16).astype(np.float32)
    # Perfect binary signal: y is completely decodable from X[:, 0]
    y = (X[:, 0] > 0).astype(float)

    # Bug pattern: circular context — mask equals the labels themselves
    circular_mask = y.astype(bool)

    evaluator = ProbeEvaluator(probe_name="linear", n_trials=5)
    results = evaluator.evaluate_context_split(X, y, circular_mask)

    in_acc = results["in_context_acc"]
    out_acc = results["out_context_acc"]
    gap = abs(in_acc - out_acc)

    # Both accuracies should be high (probe is well-trained) ...
    assert in_acc > 0.80, f"Expected high in_context_acc, got {in_acc:.3f}"
    assert out_acc > 0.80, f"Expected high out_context_acc, got {out_acc:.3f}"
    # ... but the gap should be near zero (the split is uninformative)
    assert gap < 0.15, (
        f"Circular context mask produced gap={gap:.3f}. "
        "With a circular mask the in/out accuracies should be nearly equal. "
        "If gap is large, the evaluator logic has changed — re-examine Bug C."
    )


def test_independent_context_mask_reveals_contextual_encoding():
    """Correct usage: an independent context mask exposes meaningful encoding differences.

    The concept (y) is only correlated with activations when context=True (shard active).
    A probe trained on all data will learn the correlation; evaluated on:
      - in-context samples  → high accuracy  (signal present)
      - out-of-context      → near chance    (no signal)
    This is what experiment1_probing should measure after the Bug C fix.
    """
    np.random.seed(7)
    n = 500
    X = np.random.randn(n, 16).astype(np.float32)

    # Context is defined independently of y (e.g., cheese_proximity < threshold)
    context_mask = np.zeros(n, dtype=bool)
    context_mask[:250] = True  # first 250 = "in context"

    # Concept y: perfectly decodable in-context, pure noise out-of-context
    y = np.random.randint(0, 2, n).astype(float)
    y[:250] = (X[:250, 0] > 0).astype(float)

    evaluator = ProbeEvaluator(probe_name="linear", n_trials=5)
    results = evaluator.evaluate_context_split(X, y, context_mask)

    in_acc = results["in_context_acc"]
    out_acc = results["out_context_acc"]

    assert in_acc > 0.75, (
        f"Expected in_context_acc > 0.75 when concept has signal in context, "
        f"got {in_acc:.3f}"
    )
    assert in_acc > out_acc + 0.10, (
        f"Expected in_context_acc >> out_context_acc (contextual encoding), "
        f"but got in={in_acc:.3f}, out={out_acc:.3f}"
    )


def test_bonferroni_threshold():
    evaluator = ProbeEvaluator()
    threshold = evaluator.bonferroni_threshold(n_comparisons=10, alpha=0.01)
    assert threshold == pytest.approx(0.001)


def test_probe_factory_raises_unknown():
    with pytest.raises(ValueError, match="Unknown probe"):
        ProbeFactory("nonexistent_probe")
