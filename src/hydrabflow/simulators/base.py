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

    # Two optional hooks, for a forward model that is not runnable in-process (an external
    # radiative-transfer code, an instrument pipeline). Implement them and the `simulate` stage
    # is skipped entirely; `pipeline.io.load_config_dataset` / `pipeline.adapter.build_adapter` pick
    # up. Neither is declared here as a method, so `hasattr` is the test and no existing
    # simulator changes:
    #
    #   load_dataset(self) -> Dataset
    #       The rows this simulator's forward model already produced, keyed as
    #       `parameter_names` + `observable_keys`. `sample_prior`/`simulate` may raise.
    #   build_adapter(self, cfg) -> bf.adapters.Adapter
    #       A bespoke adapter, for a layout `AdapterConfig`'s four key lists cannot express
    #       (grouped inputs, per-key transforms, conditions routed into the summary network).
    #       Still declare `parameter_names`/`observable_keys`: they drive `select_adapter_keys`.

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
        want = sorted(set(self.parameter_names) | set(self.observable_keys))
        got = {k: np.shape(v) for k, v in data.items()}
        if sorted(got) != want or any(s[:1] != (n,) for s in got.values()):
            raise ValueError(
                f"{type(self).__name__} must return {want} with leading axis n={n}, got {got}"
            )
        return data
