import os
from typing import Dict, Type, TypeVar

from src.environment_module.base_env import BaseEnv
from src.utils.auto_import import import_modules

ENV_FACTORY: Dict[str, Type[BaseEnv]] = {}

_T_Env = TypeVar("_T_Env", bound=BaseEnv)


def register_env(name: str):
    def decorator(cls: Type[_T_Env]) -> Type[_T_Env]:
        ENV_FACTORY[name] = cls
        return cls
    return decorator


def EnvFactory(name: str) -> Type[BaseEnv]:
    cls = ENV_FACTORY.get(name)
    if cls is None:
        raise ValueError(f"Unknown env: '{name}'. Available: {list(ENV_FACTORY)}")
    return cls


_dir = os.path.dirname(__file__)
for _subdir in ("maze", "minigrid"):
    _submod = os.path.join(_dir, _subdir)
    if os.path.isdir(_submod):
        import_modules(_submod, f"src.environment_module.{_subdir}")

__all__ = ["EnvFactory", "register_env", "BaseEnv", "ENV_FACTORY"]
