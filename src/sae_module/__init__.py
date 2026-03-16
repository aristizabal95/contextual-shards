import os
from typing import Dict, Type
from src.utils.auto_import import import_modules

SAE_FACTORY: Dict[str, Type] = {}


def register_sae(name: str):
    def decorator(cls):
        SAE_FACTORY[name] = cls
        return cls
    return decorator


def SAEFactory(name: str) -> Type:
    cls = SAE_FACTORY.get(name)
    if cls is None:
        raise ValueError(f"Unknown SAE: {name}. Available: {list(SAE_FACTORY)}")
    return cls


_dir = os.path.dirname(__file__)
import_modules(_dir, "src.sae_module")

from src.sae_module.model.sparse_autoencoder import SparseAutoencoder  # noqa: E402

__all__ = ["SAEFactory", "register_sae", "SparseAutoencoder"]
