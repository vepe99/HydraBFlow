"""Lotka-Volterra simulator: the differentiable testbed for simulator-gradient guidance.

Thin ``BaseSimulator`` wrapper around :mod:`hydrabflow.simulators.lv_jax`, which holds the JAX
forward model, prior and likelihood. Keeping the maths in ``lv_jax`` means the dataset written here,
the guidance gradient in ``networks.guided_diffusion``, and the MCMC reference in
``pipeline.reference`` are provably the same model — if they drifted apart the study would measure
nothing.

Parameters are the four **log**-rates (see ``lv_jax`` for why log space). The observable ``x`` has
shape ``(n, n_obs, 2)`` — prey/predator abundance at ``n_obs`` observation times — which the
``time_series_transformer`` summary network consumes directly.

Observations are written in **physical units** (raw abundances). The log transform the network needs
is a preprocessing step (``conf/preprocessing/lv.yaml``), not part of the simulator, precisely so
the guided sampler can load the untransformed ``x_obs`` that the likelihood is defined on.

Config: ``conf/simulator/lotka_volterra.yaml`` -> ``simulator.params`` (see ``lv_jax.make_config``).
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from hydrabflow.simulators import lv_jax
from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.simulators.registry import register_simulator


@register_simulator("lotka_volterra")
class LotkaVolterraSimulator(BaseSimulator):
    @property
    def parameter_names(self) -> list[str]:
        return list(lv_jax.PARAMETER_NAMES)

    @property
    def observable_keys(self) -> list[str]:
        return ["x"]

    @property
    def lv_config(self) -> lv_jax.LVConfig:
        """The validated forward-model settings (also used by the guidance and reference stages)."""
        # Rebuilt on access rather than cached: LVConfig is a cheap immutable NamedTuple, and this
        # keeps the simulator safe to construct before the config is fully resolved.
        return lv_jax.make_config(self.params)

    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        theta = lv_jax.sample_prior(n, rng, self.lv_config)  # (n, 4)
        return {name: theta[:, i : i + 1] for i, name in enumerate(lv_jax.PARAMETER_NAMES)}

    def simulate(
        self, params: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        import jax.numpy as jnp

        cfg = self.lv_config
        theta = np.concatenate(
            [np.asarray(params[name]).reshape(-1, 1) for name in lv_jax.PARAMETER_NAMES], axis=1
        )  # (n, 4)

        states = lv_jax.lv_states_batch(jnp.asarray(theta, dtype=jnp.float32), cfg)
        # The JAX key is derived from the pipeline-supplied generator, so observation noise stays
        # tied to cfg.seed and to pipeline.io's per-chunk seeding (reproducible + resumable).
        noisy = lv_jax.add_observation_noise(states, lv_jax.key_from_rng(rng), cfg)
        return {"x": np.asarray(noisy, dtype=np.float64)}

    # --- differentiability seam for gradient guidance (see BaseSimulator) ----------------------- #
    def jax_log_likelihood(self, x_obs: np.ndarray):
        """``(theta_batch) -> (batch,)`` log-likelihood of a single observation ``x_obs``.

        ``x_obs`` must be in physical units (raw abundances) with shape ``(n_obs, 2)``, i.e. the
        dataset as written by :meth:`simulate` and *not* the log-transformed/standardized version the
        network consumes. ``theta_batch`` is ``(batch, 4)`` log-rates in ``PARAMETER_NAMES`` order.
        """
        import jax.numpy as jnp

        cfg = self.lv_config
        obs = jnp.asarray(x_obs, dtype=jnp.float32)
        expected = (cfg.n_obs, 2)
        if obs.shape != expected:
            raise ValueError(f"x_obs must have shape {expected}, got {tuple(obs.shape)}")

        def log_likelihood_batch(theta):
            return lv_jax.log_likelihood_batch(theta, obs, cfg)

        return log_likelihood_batch

    def jax_theta_clip(self, n_std: float = 4.0):
        """Smooth tanh squash of the log-rates into ``prior_mean +- n_std * prior_std``."""
        cfg = self.lv_config

        def clip(theta):
            return lv_jax.soft_clip_theta(theta, cfg, n_std)

        return clip

    def jax_log_prior(self):
        """``(theta_batch) -> (batch,)`` log-prior — used by the MALA reference sampler."""
        cfg = self.lv_config

        def log_prior_batch(theta):
            import jax

            return jax.vmap(lambda t: lv_jax.log_prior(t, cfg))(theta)

        return log_prior_batch
