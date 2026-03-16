import os
from typing import Dict, Type
from src.probe_module.base_probe import BaseProbe
from src.utils.auto_import import import_modules

PROBE_FACTORY: Dict[str, Type[BaseProbe]] = {}


def register_probe(name: str):
    def decorator(cls: Type[BaseProbe]):
        PROBE_FACTORY[name] = cls
        return cls
    return decorator


def ProbeFactory(name: str) -> Type[BaseProbe]:
    cls = PROBE_FACTORY.get(name)
    if cls is None:
        raise ValueError(f"Unknown probe: {name}. Available: {list(PROBE_FACTORY)}")
    return cls


_dir = os.path.dirname(__file__)
import_modules(_dir, "src.probe_module")

__all__ = ["ProbeFactory", "register_probe", "BaseProbe"]
