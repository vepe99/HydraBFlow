"""Two Moons: the classic bimodal SBI benchmark, as a worked example simulator.

The forward model has two parameters ``theta1``, ``theta2`` (uniform prior) and produces a 2-D
observation whose posterior is famously crescent/bimodal — a good smoke test for an inference
network's ability to capture non-Gaussian, multimodal posteriors.

Generative process (one observation), following the standard `sbibm` definition::

    a ~ Uniform(-pi/2, pi/2)
    r ~ Normal(mean_radius, std_radius)
    p = (r*cos(a) + 0.25,  r*sin(a))
    x1 = p1 - |theta1 + theta2| / sqrt(2)
    x2 = p2 + (-theta1 + theta2) / sqrt(2)

The intrinsic noise ``(a, r)`` is drawn from the ``rng`` passed in by the pipeline, so the whole
simulator is stochastic yet fully reproducible from ``cfg.seed`` (see ``utils.seed``).

Observable shape: ``(n, n_obs, 2)``. ``n_obs`` is the number of i.i.d. observations generated for
each parameter draw (the "set size"). The default ``n_obs=1`` is the canonical single-observation
benchmark; raising it yields a set the SetTransformer/DeepSet summary network can pool over (and
gives the per-batch augmentations a set to act on).

Config (``conf/simulator/two_moons.yaml`` -> ``simulator.params``):
  * ``prior_low`` / ``prior_high`` — uniform prior bounds for both parameters (default -1 / 1).
  * ``n_obs`` — i.i.d. observations per parameter / summary-set size (default 1).
  * ``mean_radius`` / ``std_radius`` — intrinsic-noise radius distribution (default 0.1 / 0.01).
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.simulators.registry import register_simulator


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
