"""Two observables of different kinds, to demonstrate multi-observable fusion.

Ported from BayesFlow's ``examples/Multimodal_Data.ipynb``. Two experiments observe the same
parameters in different formats:

* ``x_set``   — ``n_set`` exchangeable 2-D draws (an unordered set)  -> a set encoder;
* ``x_series`` — a ``n_steps``-long 2-D random walk plus a time channel -> a sequence encoder.

Because ``observable_keys`` has two entries, the adapter ``group``s them and
``registry.build_summary_network`` builds one backbone per key behind a ``FusionNetwork``. The
per-key architectures are chosen in ``conf/simulator/multimodal.yaml``'s companion network config
(``model.summary_network.params.backbones``).

The time column is concatenated onto ``x_series``' features, so the sequence encoder is told where
it is with ``params.time_axis: -1``.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from hydrabflow.registry import register_simulator
from hydrabflow.simulators.base import BaseSimulator


@register_simulator("multimodal")
class MultimodalSimulator(BaseSimulator):
    parameter_names = ["mu1", "mu2", "sigma"]
    observable_keys = ["x_set", "x_series"]

    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        mu = rng.normal(size=(n, 2))
        sigma = rng.gamma(5.0, 0.1, size=(n, 1))
        return {"mu1": mu[:, 0:1], "mu2": mu[:, 1:2], "sigma": sigma}

    def simulate(
        self, theta: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        mu = np.concatenate(
            [np.asarray(theta["mu1"]).reshape(-1, 1), np.asarray(theta["mu2"]).reshape(-1, 1)],
            axis=-1,
        )[:, None, :]  # (n, 1, 2), broadcasts over the set / time axis
        sigma = np.asarray(theta["sigma"]).reshape(-1, 1, 1)
        n = mu.shape[0]

        n_set = int(self.params.get("n_set", 5))
        x_set = rng.normal(mu, sigma, size=(n, n_set, 2))

        n_steps = int(self.params.get("n_steps", 20))
        walk = np.cumsum(rng.normal(mu, sigma, size=(n, n_steps, 2)), axis=1)
        time = np.broadcast_to(np.linspace(0.0, 1.0, n_steps)[None, :, None], (n, n_steps, 1))
        x_series = np.concatenate([walk, time], axis=-1)  # (n, n_steps, 3): the last channel is t

        return {"x_set": x_set, "x_series": x_series}
