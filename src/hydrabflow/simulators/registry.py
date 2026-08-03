"""Name -> simulator class. New simulators self-register with ``@register_simulator("name")``."""

from __future__ import annotations

from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.utils.registry import Registry

SIMULATORS: Registry[type[BaseSimulator]] = Registry("simulator")
register_simulator = SIMULATORS.add


def get_simulator(cfg) -> BaseSimulator:
    """Instantiate the simulator selected by ``cfg.simulator`` (a ``SimulatorConfig``)."""
    return SIMULATORS.get(cfg.name)(params=cfg.params)
