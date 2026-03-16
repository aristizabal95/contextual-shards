import numpy as np
import pytest
from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset


def test_write_and_read_activations(tmp_path):
    path = str(tmp_path / "test.h5")
    with HDF5ActivationDataset(path, mode="w") as ds:
        acts = {"block1": np.random.randn(10, 16, 8, 8).astype(np.float32)}
        labels = {"cheese_presence": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)}
        ds.write_batch(activations=acts, labels=labels)

    with HDF5ActivationDataset(path, mode="r") as ds2:
        assert len(ds2) == 10
        sample = ds2[0]
        assert "block1" in sample["activations"]
        assert "cheese_presence" in sample["labels"]
        assert sample["activations"]["block1"].shape == (16, 8, 8)


def test_incremental_write(tmp_path):
    path = str(tmp_path / "incremental.h5")
    with HDF5ActivationDataset(path, mode="w") as ds:
        acts1 = {"fc": np.random.randn(5, 256).astype(np.float32)}
        labels1 = {"cheese_presence": np.ones(5, dtype=np.float32)}
        ds.write_batch(activations=acts1, labels=labels1)

        acts2 = {"fc": np.random.randn(3, 256).astype(np.float32)}
        labels2 = {"cheese_presence": np.zeros(3, dtype=np.float32)}
        ds.write_batch(activations=acts2, labels=labels2)

        assert len(ds) == 8


def test_get_all_activations(tmp_path):
    path = str(tmp_path / "all.h5")
    with HDF5ActivationDataset(path, mode="w") as ds:
        acts = {"fc": np.random.randn(20, 256).astype(np.float32)}
        labels = {"cheese_presence": np.ones(20, dtype=np.float32)}
        ds.write_batch(activations=acts, labels=labels)

    with HDF5ActivationDataset(path, mode="r") as ds:
        all_acts = ds.get_all_activations("fc")
        assert all_acts.shape == (20, 256)
