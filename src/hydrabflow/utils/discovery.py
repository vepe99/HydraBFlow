"""Auto-import the modules of a package so ``@register_*`` decorators run.

The registries (simulators, preprocessing steps, augmentations, network builders) are filled by
import side effects. This helper imports every module directly inside a package, so dropping a
new file into ``simulators/``, ``preprocessing/``, ``augmentation/`` or ``networks/`` registers
its components automatically — no ``__init__.py`` edit needed.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable


def import_submodules(package_name: str, package_path: Iterable[str]) -> None:
    """Import every non-underscore module directly inside a package.

    Call from a package's ``__init__.py`` as ``import_submodules(__name__, __path__)``.
    Modules whose names start with ``_`` are skipped so private helpers stay private.
    """
    for info in pkgutil.iter_modules(list(package_path)):
        if not info.name.startswith("_"):
            importlib.import_module(f"{package_name}.{info.name}")
