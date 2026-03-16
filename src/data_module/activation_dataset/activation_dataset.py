import h5py
import numpy as np
from typing import Any, Dict
from torch.utils.data import Dataset


class HDF5ActivationDataset(Dataset):
    """Stores and retrieves neural network activations and concept labels from HDF5.

    Supports incremental writing via write_batch() and random-access reading via __getitem__.
    """

    def __init__(self, path: str, mode: str = "r"):
        self.path = path
        self.mode = mode
        self._file = h5py.File(path, mode)
        self._length: int = 0

        if mode == "w":
            self._file.create_group("activations")
            self._file.create_group("labels")
        else:
            act_keys = list(self._file["activations"].keys())
            if act_keys:
                self._length = len(self._file["activations"][act_keys[0]])

    def write_batch(
        self,
        activations: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
    ) -> None:
        """Append a batch of activations and labels to the dataset."""
        if not activations:
            return
        n = next(iter(activations.values())).shape[0]

        for layer_name, data in activations.items():
            key = f"activations/{layer_name}"
            if key in self._file:
                current_size = self._file[key].shape[0]
                self._file[key].resize(current_size + n, axis=0)
                self._file[key][current_size:] = data
            else:
                maxshape = (None,) + data.shape[1:]
                self._file.create_dataset(key, data=data, maxshape=maxshape, chunks=True)

        for label_name, data in labels.items():
            key = f"labels/{label_name}"
            if key in self._file:
                current_size = self._file[key].shape[0]
                self._file[key].resize(current_size + n, axis=0)
                self._file[key][current_size:] = data
            else:
                self._file.create_dataset(
                    key, data=data, maxshape=(None,) + data.shape[1:], chunks=True
                )

        self._length += n

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        acts = {
            k: self._file[f"activations/{k}"][idx]
            for k in self._file["activations"]
        }
        labs = {
            k: self._file[f"labels/{k}"][idx]
            for k in self._file["labels"]
        }
        return {"activations": acts, "labels": labs}

    def get_all_activations(self, layer: str) -> np.ndarray:
        """Load all activations for a single layer into memory."""
        return self._file[f"activations/{layer}"][:]

    def get_all_labels(self, concept: str) -> np.ndarray:
        """Load all labels for a single concept into memory."""
        return self._file[f"labels/{concept}"][:]

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
