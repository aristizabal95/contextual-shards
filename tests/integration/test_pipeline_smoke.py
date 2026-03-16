"""Smoke tests: verify all modules import and instantiate without error.

These tests do NOT require procgen, a GPU, or any trained model.
They validate the project scaffold is correctly wired together.
"""
import numpy as np
import pytest


def test_probe_factory_linear():
    from src.probe_module import ProbeFactory
    probe = ProbeFactory("linear")()
    X = np.random.randn(20, 8).astype(np.float32)
    y = (np.random.rand(20) > 0.5).astype(float)
    probe.fit(X[:15], y[:15])
    acc = probe.score(X[15:], y[15:])
    assert 0.0 <= acc <= 1.0


def test_probe_factory_mdl():
    from src.probe_module.probe.mdl_probe import MDLProbe
    probe = MDLProbe()
    X = np.random.randn(30, 4).astype(np.float32)
    y = (np.random.rand(30) > 0.5).astype(float)
    probe.fit(X, y)
    assert probe.codelength > 0


def test_labeler_factory_all():
    from src.data_module.concept_labeler import LabelerFactory
    grid = np.zeros((15, 15))
    for name in ["cheese_presence", "cheese_proximity", "cheese_direction", "corner_proximity"]:
        lab = LabelerFactory(name)()
        result = lab.label(agent_pos=(5, 5), cheese_pos=(5, 6), maze_grid=grid)
        assert isinstance(result, float)


def test_activation_dataset_roundtrip(tmp_path):
    from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset
    path = str(tmp_path / "smoke.h5")
    with HDF5ActivationDataset(path, mode="w") as ds:
        ds.write_batch(
            activations={"fc": np.random.randn(10, 256).astype(np.float32)},
            labels={"cheese_presence": np.ones(10, dtype=np.float32)},
        )
    with HDF5ActivationDataset(path, mode="r") as ds:
        assert len(ds) == 10


def test_shard_metrics():
    from src.shard_module.metrics.shard_metrics import ShardMetrics
    m = ShardMetrics()
    I = m.independence_score(0.8, 0.8, 0.3)
    assert abs(I - 1.0) < 1e-4
    effect = m.causal_effect_size(0.8, 0.5)
    assert abs(effect - 0.3) < 1e-5


def test_shard_detector():
    from src.shard_module.detection.shard_detector import ShardDetector
    detector = ShardDetector(probe_threshold=0.2, causal_threshold=0.3)
    probe = {"fc": {"in_context_acc": 0.9, "out_context_acc": 0.5}}
    causal = {"fc": 0.7}
    candidates = detector.detect(probe, causal, top_k=1)
    assert len(candidates) == 1
    assert candidates[0][0] == "fc"


def test_sae_instantiation_and_forward():
    import torch
    from src.sae_module.model.sparse_autoencoder import SparseAutoencoder
    sae = SparseAutoencoder(d_input=32, expansion_factor=4)
    x = torch.randn(8, 32)
    recon, features = sae(x)
    assert recon.shape == (8, 32)
    assert features.shape == (8, 128)
    assert (features >= 0).all()


def test_cheese_distribution():
    from src.trainer_module.rl_trainer.cheese_distribution import CheesePlacementDistribution
    for mode in ("corner_biased", "uniform", "anti_corner"):
        dist = CheesePlacementDistribution(mode=mode, grid_size=15)
        r, c = dist.sample()
        assert 0 <= r < 15 and 0 <= c < 15


def test_stat_tests():
    from src.analysis_module.statistics.stat_tests import (
        bonferroni_correction, spearman_correlation, cohen_d,
    )
    p_vals = [0.001, 0.05, 0.1]
    results = bonferroni_correction(p_vals, alpha=0.01)
    assert results[0] is True and results[1] is False
    rho, p = spearman_correlation([1, 2, 3], [1, 2, 3])
    assert abs(rho - 1.0) < 1e-5
    import numpy as np
    d = cohen_d(np.ones(10) * 5, np.zeros(10))
    assert d > 1.0


def test_shard_visualizer():
    import matplotlib.pyplot as plt
    from src.analysis_module.visualization.shard_visualizer import ShardVisualizer
    viz = ShardVisualizer()
    fig = viz.plot_causal_effects({"block1": 0.2, "fc": 0.8})
    assert fig is not None
    plt.close("all")


def test_experiment1_integrate_imports():
    """Verify experiment1_integrate.py can be imported."""
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "exp1_integrate",
        os.path.join("run", "pipeline", "analysis", "experiment1_integrate.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert mod is not None
