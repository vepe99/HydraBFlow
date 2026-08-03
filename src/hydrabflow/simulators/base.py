"""Base interface every forward model implements.

A simulator is the ONLY piece a new user must write (plus a matching ``conf/simulator`` YAML).
It samples parameters from the prior and maps them to observables. Everything downstream
(dataset generation, adapter, training, evaluation) is driven by ``parameter_names`` and
``observable_keys`` and never needs to change.

Convention for shapes (batched, leading axis = number of simulations ``n``):
  * ``sample_prior(n, rng)`` -> ``{param_name: array of shape (n, 1)}``
  * ``simulate(params, rng)`` -> ``{observable_key: array of shape (n, *event_shape)}``

The dataset written to disk is the union of both dicts, so each ``.npz`` row is one
(parameters, observation) pair.
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
        # `params` is the free-form `simulator.params` mapping from config.
        self.params: Dict[str, Any] = dict(params or {})

    @abstractmethod
    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Draw ``n`` prior samples. Returns ``{param_name: (n, 1)}``."""

    @abstractmethod
    def simulate(
        self, params: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        """Run the forward model on a batch of parameters. Returns ``{observable_key: (n, ...)}``."""

    # --------------------------------------------------------------------------------------- #
    # Convenience: one call producing a full dataset chunk (parameters + observables merged).
    # Infrastructure (pipeline.simulate) uses this; subclasses normally need not override it.
    # --------------------------------------------------------------------------------------- #
    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        params = self.sample_prior(n, rng)
        observables = self.simulate(params, rng)
        return {**params, **observables}
