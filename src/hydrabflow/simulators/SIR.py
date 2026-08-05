from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.simulators.registry import register_simulator


def convert_params(mu, phi):
    """Helper function to convert mean/dispersion parameterization of a negative binomial to N and p,
    as expected by numpy's negative_binomial.

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """

    r = phi
    var = mu + 1 / r * mu**2
    p = (var - mu) / var
    return r, 1 - p


@register_simulator("sir")
class StationarySIR(BaseSimulator):
    parameter_name = ['lambda', 'mu', 'D', 'I0', 'psi']
    observable_keys = ['cases']

    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        lambd = rng.lognormal(mean=np.log(0.4), sigma=0.5, size=(n, 1))
        mu = rng.lognormal(mean=np.log(1 / 8), sigma=0.2, size=(n, 1))
        D = rng.lognormal(mean=np.log(8), sigma=0.2, size=(n, 1))
        I0 = rng.gamma(shape=2, scale=20, size=(n, 1))
        psi = rng.exponential(5, size=(n, 1))
        return {"lambd": lambd, "mu": mu, "D": D, "I0": I0, "psi": psi}