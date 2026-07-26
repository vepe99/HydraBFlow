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
    # Optional: differentiability seam for gradient guidance.
    #
    # A simulator with a tractable, differentiable likelihood can opt in by implementing the two
    # methods below. That makes it usable with the `guided_diffusion` inference network, which
    # injects `grad log p(x_obs | theta)` into the reverse diffusion process (see
    # `networks.guided_diffusion` and `pipeline.guidance`). Simulators that cannot do this are
    # unaffected — the default raises only if guidance is actually requested.
    # --------------------------------------------------------------------------------------- #
    def jax_log_likelihood(self, x_obs: np.ndarray):
        """Return a JAX-traceable ``(theta_batch) -> (batch,)`` log-likelihood for a fixed ``x_obs``.

        ``theta_batch`` has shape ``(batch, len(parameter_names))`` in the **same order and the same
        (physical) units** as ``parameter_names`` — that ordering is what the adapter concatenates
        into ``inference_variables``, so it is also the layout of the diffusion state.

        ``x_obs`` is a single observation in **physical units** (i.e. as written by ``simulate``,
        before any preprocessing), with shape ``event_shape``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement jax_log_likelihood, so it cannot be used "
            "with gradient guidance. Implement it (a JAX-traceable log-likelihood of a fixed "
            "observation, vectorized over a batch of parameter vectors) to opt in."
        )

    def jax_gaussian_observation_model(self, x_obs: np.ndarray):
        """Optional: expose the forward model as a **Gaussian** observation model.

        Required only by uncertainty-inflated (PiGDM-style) guidance, which cannot work from a scalar
        log-likelihood: it needs the *Jacobian* of the forward map, so it can inflate the observation
        covariance by the diffusion model's current denoising uncertainty. See
        ``networks.guided_diffusion.GuidedDiffusionModel.guidance_pigdm``.

        Return ``(forward_log, observation_log, obs_variance)`` describing the model

            observation_log = forward_log(theta) + N(0, obs_variance)

        where ``forward_log`` maps a **single** parameter vector ``(n_params,)`` to the noise-free
        transformed observation ``(n_data,)``, and ``observation_log`` is the same transform applied to
        ``x_obs``. "log" in the names reflects that the transform is usually a log for multiplicative
        noise; any transform that makes the noise additive-Gaussian is valid.

        ``forward_log`` must be JAX-traceable and differentiable, and must accept one parameter vector
        (it is ``vmap``-ed and ``jacfwd``-ed by the caller).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement jax_gaussian_observation_model, so it cannot "
            "be used with precondition='pigdm'. Implement it (forward_log, observation_log, "
            "obs_variance) to opt in, or use precondition='none'."
        )

    def jax_theta_clip(self):
        """Optional smooth squash applied to parameters before the likelihood, or ``None``.

        Guidance evaluates the likelihood at the diffusion model's *denoised estimate*, which early
        in the reverse process is far outside the prior. A smooth (differentiable) squash into a
        plausible region keeps the forward model numerically sane there. Return ``None`` for no clip.
        """
        return None
