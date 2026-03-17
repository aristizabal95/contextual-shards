import os
from typing import Dict, Type, TypeVar

from src.agent_module.base_agent import BaseAgent
from src.utils.auto_import import import_modules

AGENT_FACTORY: Dict[str, Type[BaseAgent]] = {}

_T_Agent = TypeVar("_T_Agent", bound=BaseAgent)


def register_agent(name: str):
    def decorator(cls: Type[_T_Agent]) -> Type[_T_Agent]:
        AGENT_FACTORY[name] = cls
        return cls
    return decorator


def AgentFactory(name: str) -> Type[BaseAgent]:
    cls = AGENT_FACTORY.get(name)
    if cls is None:
        raise ValueError(f"Unknown agent: '{name}'. Available: {list(AGENT_FACTORY)}")
    return cls


_dir = os.path.dirname(__file__)
for _subdir in ("policy", "hooks"):
    _submod = os.path.join(_dir, _subdir)
    if os.path.isdir(_submod):
        import_modules(_submod, f"src.agent_module.{_subdir}")

__all__ = ["AgentFactory", "register_agent", "BaseAgent", "AGENT_FACTORY"]
