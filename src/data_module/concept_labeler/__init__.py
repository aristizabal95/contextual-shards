import os
from typing import Dict, Type
from src.data_module.concept_labeler.base_labeler import BaseConceptLabeler
from src.utils.auto_import import import_modules

LABELER_FACTORY: Dict[str, Type[BaseConceptLabeler]] = {}

def register_labeler(name: str):
    def decorator(cls: Type[BaseConceptLabeler]):
        LABELER_FACTORY[name] = cls
        return cls
    return decorator

def LabelerFactory(name: str) -> Type[BaseConceptLabeler]:
    cls = LABELER_FACTORY.get(name)
    if cls is None:
        raise ValueError(f"Unknown labeler: {name}. Available: {list(LABELER_FACTORY)}")
    return cls

_dir = os.path.dirname(__file__)
import_modules(_dir, "src.data_module.concept_labeler")

__all__ = ["LabelerFactory", "register_labeler", "BaseConceptLabeler"]
