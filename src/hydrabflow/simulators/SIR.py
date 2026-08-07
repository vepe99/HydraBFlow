################################################################################################################
### This simulator is copied from https://bayesflow.org/v2.0.13/_examples/SIR_Posterior_Estimation.html ########
################################################################################################################


from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.registry import register_simulator


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
    parameter_names = ['lambd', 'mu', 'D', 'I0', 'psi']
    observable_keys = ['cases']
    # The reporting delay D sets the number of timesteps, so the forward model cannot vectorize:
    # BaseSimulator.sample calls simulate() once per draw and stacks the results.
    is_batched = False

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.total_population = float(self.params.get("tot_pop"))
        self.T = int(self.params.get("T"))
        self.eps = float(self.params.get("eps"))

    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        lambd = rng.lognormal(mean=np.log(self.params.get("lambd_prior")["mean"]), sigma=self.params.get("lambd_prior")["sigma"], size=(n, 1))
        mu = rng.lognormal(mean=np.log(1 / self.params.get("mu_prior")["inverse_mean"]), sigma=self.params.get("mu_prior")["sigma"], size=(n, 1))
        D = rng.lognormal(mean=np.log(self.params.get("D_prior")["mean"]), sigma=self.params.get("D_prior")["sigma"], size=(n, 1))
        I0 = rng.gamma(shape=self.params.get("I0_prior")["shape"], scale=self.params.get("I0_prior")["scale"], size=(n, 1))
        psi = rng.exponential(scale=self.params.get("psi_prior")["scale"], size=(n, 1))
        return {"lambd": lambd, "mu": mu, "D": D, "I0": I0, "psi": psi}
    
    def simulate(self, theta: Mapping[str, np.ndarray], rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """One draw: ``theta`` values are shape ``(1,)``, returns ``cases`` of shape ``(T, 1)``."""
        lambd, mu, psi = theta["lambd"].item(), theta["mu"].item(), theta["psi"].item()
        I0 = np.ceil(theta["I0"]).item()
        D = int(np.round(theta["D"]).item())

        # Only S and I feed back; the recovered compartment is never read, so it isn't tracked.
        S, I, C = self.total_population - I0, I0, [I0]
        for _ in range(1, self.T + D):
            I_new = lambd * I * S / self.total_population
            S -= I_new
            I = np.clip(I + I_new - mu * I, 0.0, self.total_population)
            C.append(I_new)

        # C holds T + D entries, so dropping the first D reported-delay steps always leaves T.
        r, p = convert_params(np.clip(C[D:], 0, self.total_population) + self.eps, psi)
        return dict(cases=rng.negative_binomial(r, p)[:, None].astype(np.float64))  # (T, 1)


