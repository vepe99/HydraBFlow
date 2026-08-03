"""One name -> component registry, shared by every extension point.

Simulators, preprocessing steps, augmentations, and network builders all resolve by name from an
instance of :class:`Registry`. :func:`discover` imports a package's modules so the decorators run,
which is why dropping a file into ``simulators/`` is enough to register what it defines.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict, Generic, Iterable, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection filled by ``@registry.add("name")`` decorators."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.items: Dict[str, T] = {}

    def add(self, name: str) -> Callable[[T], T]:
        def _wrap(obj: T) -> T:
            if self.items.setdefault(name, obj) is not obj:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            return obj

        return _wrap

    def get(self, name: str) -> T:
        if name not in self.items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Registered: {sorted(self.items)}. "
                f"To add one, drop a module in the package and decorate it."
            )
        return self.items[name]


def discover(package_name: str, package_path: Iterable[str]) -> None:
    """Import every non-underscore module in a package, so its decorators run."""
    for info in pkgutil.iter_modules(list(package_path)):
        if not info.name.startswith("_"):
            importlib.import_module(f"{package_name}.{info.name}")
