"""Base interface every forward model implements.

A simulator is the only piece a new user must write (plus a ``conf/simulator/<name>.yaml``). Shapes
are batched, leading axis = number of simulations ``n``:

  * ``sample_prior(n, rng)`` -> ``{param_name: (n, 1)}``
  * ``simulate(params, rng)`` -> ``{observable_key: (n, *event_shape)}``

The dataset on disk is the union of both dicts, so each row is one (parameters, observation) pair.
All randomness must come from the passed ``rng``, so runs reproduce from ``cfg.seed``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping

import numpy as np


class BaseSimulator(ABC):
    """Abstract forward model. Subclass + register via ``@register_simulator``."""

    #: Ordered names of the inferred parameters (become ``inference_variables``).
    parameter_names: list[str] = []
    #: Keys of the observable arrays. One key = single observable; >1 enables fusion.
    observable_keys: list[str] = []

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        # The free-form `simulator.params` mapping from config.
        self.params: Dict[str, Any] = dict(params or {})

    @abstractmethod
    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Draw ``n`` prior samples. Returns ``{param_name: (n, 1)}``."""

    @abstractmethod
    def simulate(
        self, params: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        """Run the forward model on a batch of parameters. Returns ``{observable_key: (n, ...)}``."""

    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """One dataset chunk: prior draws merged with their observables. Rarely overridden."""
        params = self.sample_prior(n, rng)
        return {**params, **self.simulate(params, rng)}
