import os
import importlib

def import_modules(directory: str, package: str) -> None:
    """Auto-import all Python modules in a directory for registry side-effects."""
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".py") and not filename.startswith("_"):
            module_name = filename[:-3]
            importlib.import_module(f"{package}.{module_name}")
        elif os.path.isdir(os.path.join(directory, filename)):
            subdir = os.path.join(directory, filename)
            subpackage = f"{package}.{filename}"
            if os.path.exists(os.path.join(subdir, "__init__.py")):
                importlib.import_module(subpackage)
