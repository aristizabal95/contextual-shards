import torch
import numpy as np
import pytest
from src.causal_module.shard_vector.shard_vector import ShardVector


def test_shard_vector_mean_diff_torch():
    sv = ShardVector()
    context = {"block1": torch.ones(10, 16, 4, 4) * 2.0}
    baseline = {"block1": torch.ones(10, 16, 4, 4) * 1.0}
    vectors = sv.compute(context, baseline)
    assert "block1" in vectors
    assert vectors["block1"].shape == (16, 4, 4)
    assert torch.allclose(vectors["block1"], torch.ones(16, 4, 4))


def test_shard_vector_mean_diff_numpy():
    sv = ShardVector()
    context = {"fc": np.random.randn(50, 256).astype(np.float32) + 1.0}
    baseline = {"fc": np.random.randn(50, 256).astype(np.float32)}
    vectors = sv.compute(context, baseline)
    assert "fc" in vectors
    assert vectors["fc"].shape == (256,)


def test_shard_vector_normalize():
    sv = ShardVector()
    context = {"fc": torch.ones(5, 4) * 3.0}
    baseline = {"fc": torch.zeros(5, 4)}
    vectors = sv.compute(context, baseline)
    normalized = sv.normalize(vectors)
    norm = normalized["fc"].norm().item()
    assert abs(norm - 1.0) < 1e-5


def test_shard_vector_multiple_layers():
    sv = ShardVector()
    context = {
        "block1": torch.randn(20, 16, 8, 8),
        "fc": torch.randn(20, 256),
    }
    baseline = {
        "block1": torch.randn(20, 16, 8, 8),
        "fc": torch.randn(20, 256),
    }
    vectors = sv.compute(context, baseline)
    assert set(vectors.keys()) == {"block1", "fc"}
