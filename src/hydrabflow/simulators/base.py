"""Base interface every forward model implements.

A simulator is the only piece a new user must write (plus a ``conf/simulator/<name>.yaml``). Shapes
are batched, leading axis = number of simulations ``n``:

  * ``sample_prior(n, rng)`` -> ``{param_name: (n, 1)}``
  * ``simulate(theta, rng)`` -> ``{observable_key: (n, *event_shape)}``

Set ``is_batched = False`` when the forward model cannot vectorize (a parameter that sets a loop
length, an ODE solver, an external binary): ``simulate`` then gets one draw ``{param_name: (1,)}``
and returns ``{observable_key: event_shape}``, and ``sample`` loops and stacks.

``theta`` is the prior draw(s); ``self.params`` is the free-form ``simulator.params`` config
mapping (prior bounds, set sizes). Two different things, deliberately two different names.

The dataset on disk is the union of both dicts, so each row is one (parameters, observation) pair.
A simulator may return *extra* arrays beyond its declared names (fixed constants, diagnostics,
context keys); the adapter drops what it does not use. All randomness must come from the passed
``rng``, so runs reproduce from ``cfg.seed``.

Hierarchical (compositional) simulators additionally split their parameters into *global* ones,
shared by a group of exchangeable observations (one galactic potential, several streams), and
*local* per-member ones, and implement ``sample_compositional``. The flat defaults below mean a
single-level simulator needs none of that.
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
    #: False -> ``simulate`` handles one draw at a time and ``sample`` loops and stacks.
    is_batched: bool = True

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        # The free-form `simulator.params` mapping from config.
        self.params: Dict[str, Any] = dict(params or {})

    @abstractmethod
    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Draw ``n`` prior samples. Returns ``{param_name: (n, 1)}``."""

    @abstractmethod
    def simulate(
        self, theta: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        """Batched: ``{param_name: (n, 1)}`` -> ``{observable_key: (n, *event_shape)}``.
        With ``is_batched = False``: one draw ``{param_name: (1,)}`` -> ``{observable_key: shape}``.
        """

    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """One dataset chunk: prior draws merged with their observables. Rarely overridden."""
        theta = self.sample_prior(n, rng)
        if self.is_batched:
            obs = self.simulate(theta, rng)
        else:
            rows = [self.simulate({k: v[i] for k, v in theta.items()}, rng) for i in range(n)]
            obs = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
        data = {**theta, **obs}

        # run_chunked concatenates chunks on axis 0 and the adapter looks up the declared names, so
        # a mis-named or non-row-major return would land on disk as a plausible-looking corrupt file.
        want = set(self.parameter_names) | set(self.observable_keys)
        got = {k: np.shape(v) for k, v in data.items()}
        missing = sorted(want - set(got))
        bad_axis = sorted(k for k, s in got.items() if s[:1] != (n,))
        if missing or bad_axis:
            raise ValueError(
                f"{type(self).__name__} must return {sorted(want)} with leading axis n={n}; "
                f"missing={missing}, wrong leading axis={bad_axis}, got {got}"
            )
        return data

    # --- hierarchical (compositional) seam -------------------------------------------------- #

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

        Shape convention (``m`` = group members): globals ``(n, 1)``; locals / context keys
        ``(n, m, 1)``; observables ``(n, m, *event_shape)`` (member-independent observables may
        stay ``(n, *event_shape)``). Used by ``simulate_multistream`` for compositional test sets.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sample_compositional(); it is a "
            "single-level simulator. Hierarchical simulators must override it."
        )
