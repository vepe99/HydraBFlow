"""Two Moons: the classic bimodal SBI benchmark. Copy this file for your own forward model.

Generative process for one observation (the standard `sbibm` definition)::

    a ~ Uniform(-pi/2, pi/2);  r ~ Normal(mean_radius, std_radius)
    x1 = r*cos(a) + 0.25 - |theta1 + theta2| / sqrt(2)
    x2 = r*sin(a)        + (-theta1 + theta2) / sqrt(2)

Observable shape ``(n, n_obs, 2)``: ``n_obs`` i.i.d. observations per parameter draw, i.e. the set
size the summary network pools over. ``simulator.params`` knobs: ``prior_low``/``prior_high``,
``n_obs``, ``mean_radius``/``std_radius``.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.registry import register_simulator


@register_simulator("two_moons")
class TwoMoonsSimulator(BaseSimulator):
    parameter_names = ["theta1", "theta2"]
    observable_keys = ["x"]

    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        low = float(self.params.get("prior_low", -1.0))
        high = float(self.params.get("prior_high", 1.0))
        theta = rng.uniform(low, high, size=(n, 2))
        return {"theta1": theta[:, 0:1], "theta2": theta[:, 1:2]}

    def simulate(
        self, params: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        theta1 = np.asarray(params["theta1"]).reshape(-1, 1)  # (n, 1)
        theta2 = np.asarray(params["theta2"]).reshape(-1, 1)  # (n, 1)
        shape = (theta1.shape[0], int(self.params.get("n_obs", 1)))

        # Intrinsic noise (the only stochasticity) — drawn from the provided rng.
        a = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=shape)
        r = rng.normal(
            float(self.params.get("mean_radius", 0.1)),
            float(self.params.get("std_radius", 0.01)),
            size=shape,
        )

        p1 = r * np.cos(a) + 0.25
        p2 = r * np.sin(a)

        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        x1 = p1 - np.abs(theta1 + theta2) * inv_sqrt2  # theta (n,1) broadcasts over n_obs
        x2 = p2 + (-theta1 + theta2) * inv_sqrt2

        x = np.stack([x1, x2], axis=-1)  # (n, n_obs, 2)
        return {"x": x.astype(np.float64)}
