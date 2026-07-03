"""Simulators (forward models).

Importing this package auto-imports every module in it, so any ``@register_simulator("name")``-
decorated :class:`BaseSimulator` subclass self-registers. To add your own simulator, just drop a
new module in this folder — no need to edit this file.
"""

from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.simulators.registry import get_simulator, register_simulator
from hydrabflow.utils.discovery import import_submodules

import_submodules(__name__, __path__)  # registers skeleton, two_moons, and any user simulators

__all__ = ["BaseSimulator", "get_simulator", "register_simulator"]
