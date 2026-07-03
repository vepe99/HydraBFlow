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

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        # `params` is the free-form `simulator.params` mapping from config.
        self.params: Dict[str, Any] = dict(params or {})

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Ordered names of the inferred parameters (become ``inference_variables``)."""

    @property
    @abstractmethod
    def observable_keys(self) -> list[str]:
        """Keys of the observable arrays. One key = single observable; >1 enables fusion."""

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

    # --------------------------------------------------------------------------------------- #
    # Hierarchical (compositional) seam. A hierarchical simulator has *global* parameters
    # shared by a group of exchangeable observations (e.g. one galactic potential constraining
    # several stellar streams) and *local* parameters specific to each group member. The flat
    # defaults below mean existing single-level simulators need no changes; a hierarchical
    # simulator overrides the three properties and `sample_compositional`.
    # --------------------------------------------------------------------------------------- #

    @property
    def global_parameter_names(self) -> list[str]:
        """Inferred parameters shared across group members. Default: all parameters (flat)."""
        return list(self.parameter_names)

    @property
    def local_parameter_names(self) -> list[str]:
        """Inferred per-member parameters. Empty for single-level simulators."""
        return []

    @property
    def context_keys(self) -> list[str]:
        """Dataset keys that condition inference but are neither inferred parameters nor
        observables (e.g. the member index identifying which stream an observation is)."""
        return []

    def sample_compositional(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Draw ``n`` grouped datasets: one shared global draw + one local draw per member.

        Shape convention (``m`` = number of group members):
          * global parameters: ``(n, 1)``
          * local parameters / context keys: ``(n, m, 1)``
          * observables: ``(n, m, *event_shape)`` (member-independent observables may stay
            ``(n, *event_shape)``)

        Used by the ``simulate_multistream`` stage to build compositional test sets.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sample_compositional(); it is a "
            "single-level simulator. Hierarchical simulators must override it."
        )
