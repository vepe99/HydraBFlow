"""Simulators (forward models).

Importing this package imports the shipped simulators so they self-register in the registry.
Add your own by creating a module here with an ``@register_simulator("name")``-decorated subclass
of :class:`BaseSimulator`, then import it here (or rely on it being imported by your config).
"""

from hydraflow.simulators import skeleton  # noqa: F401  (self-registers "skeleton")
from hydraflow.simulators import two_moons  # noqa: F401  (self-registers "two_moons")
from hydraflow.simulators.base import BaseSimulator
from hydraflow.simulators.registry import get_simulator, register_simulator

__all__ = ["BaseSimulator", "get_simulator", "register_simulator"]
