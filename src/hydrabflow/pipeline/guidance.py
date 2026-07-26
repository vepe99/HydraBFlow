"""Wiring between a differentiable simulator and the guided diffusion sampler.

Keeps the model-specific parts (which likelihood, which parameter space) in the simulator and the
sampling-time mechanics (which network, which observation, which knobs) here, so
``networks.guided_diffusion`` stays generic.

The **space** question is the subtle one and is resolved here:

* the diffusion state is the adapter's ``inference_variables``, i.e. the simulator's
  ``parameter_names`` concatenated in declared order (``pipeline/adapter.py``), optionally passed
  through BayesFlow's internal standardizer (``training.standardize``);
* the likelihood is defined on **physical** parameters and a **physical** (un-preprocessed)
  observation.

:func:`attach_guidance` therefore composes ``untransform`` (undo standardization, from
``networks.guided_diffusion.build_theta_untransform``) with the simulator's own clip and likelihood,
all inside the autodiff trace, so the chain rule back to the diffusion state is automatic. The raw
observation must be handed in by the caller — the preprocessed copy the network conditions on is the
wrong input for the likelihood.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from hydrabflow.utils.logging import get_logger

log = get_logger(__name__)

#: ``eval.guidance`` / ``inference.guidance`` keys that may override the trained network's defaults.
GUIDANCE_OVERRIDE_KEYS = (
    "guidance_strength",
    "t_on",
    "t_full",
    "scaling",
    "max_grad_norm",
    "clip_theta_std",
    "skip_outside_window",
    "guidance_particles",
    "guidance_reduce",
    "particle_width",
    "particle_data_std",
    "particle_seed",
    "guidance_point",
)


def get_guided_network(workflow) -> Any:
    """Return the inference network, checking it supports guidance."""
    net = workflow.approximator.inference_network
    if not hasattr(net, "set_guidance_target"):
        raise TypeError(
            f"inference network {type(net).__name__} does not support guidance. Use "
            "model/inference_network=guided_diffusion (see conf/model/lv_guided.yaml)."
        )
    return net


def apply_guidance_overrides(net, overrides: Optional[Mapping[str, Any]]) -> dict:
    """Override the trained network's guidance knobs at sampling time (in place).

    Guidance is a sampling-time modification, so a single trained network serves a whole sweep —
    this is what makes ``eval.guidance`` meaningful without retraining. Returns the settings
    actually in force, for logging / provenance.
    """
    from omegaconf import OmegaConf

    if overrides is not None:
        if OmegaConf.is_config(overrides):
            overrides = OmegaConf.to_container(overrides, resolve=True)
        unknown = set(overrides) - set(GUIDANCE_OVERRIDE_KEYS)
        if unknown:
            raise ValueError(
                f"unknown guidance override(s) {sorted(unknown)}; "
                f"valid keys are {list(GUIDANCE_OVERRIDE_KEYS)}"
            )
        for key, value in overrides.items():
            current = getattr(net, key)
            setattr(net, key, type(current)(value) if not isinstance(value, str) else value)

    settings = {key: getattr(net, key) for key in GUIDANCE_OVERRIDE_KEYS}
    # Re-validate: the constructor's invariant can be broken by an override.
    if not 0.0 <= net.t_full <= net.t_on <= 1.0:
        raise ValueError(f"require 0 <= t_full <= t_on <= 1, got {settings}")
    return settings


def attach_guidance(workflow, simulator, x_obs: np.ndarray, net=None) -> Any:
    """Point the guided sampler at one observation ``x_obs`` (in **physical** units).

    Must be called *after* ``load_approximator``: ``pipeline.evaluate`` replaces
    ``workflow.approximator`` wholesale, which would discard a target attached earlier.
    """
    from hydrabflow.networks.guided_diffusion import GuidanceTarget, build_theta_untransform

    net = net if net is not None else get_guided_network(workflow)
    clip = simulator.jax_theta_clip(net.clip_theta_std)
    target = GuidanceTarget(
        log_likelihood_batch=simulator.jax_log_likelihood(x_obs),
        untransform=build_theta_untransform(workflow.approximator),
        clip=clip,
    )
    net.set_guidance_target(target)
    return net


def detach_guidance(net) -> None:
    """Clear the guidance target, making the network behave as a plain ``DiffusionModel``."""
    net.set_guidance_target(None)


def resolve_conditions(approximator, conditions: Optional[Mapping[str, np.ndarray]], num_samples: int):
    """Conditions tensor as the inference network sees it, repeated to ``num_samples`` rows.

    Replicates what ``ContinuousApproximator.sample`` does internally (adapter -> summary network ->
    standardizer -> repeat), which the eager diagnostic loop in ``pipeline.diagnose_guidance`` needs
    in order to call ``network.score`` directly. Returns ``None`` for an unconditional model.
    """
    import keras

    if conditions is None:
        return None
    # Private helper, but it is the only way to reproduce the exact conditioning tensor the sampler
    # builds; going through the public `sample` would run the whole integrator we want to unroll.
    resolved, _, _ = approximator._prepare_conditions(conditions, batch_size=None)
    if resolved is None:
        return None
    n_rows = int(keras.ops.shape(resolved)[0])
    if n_rows != 1:
        raise ValueError(
            f"resolve_conditions expects a single observation, got {n_rows} rows. The guided "
            "sampler is driven one observation at a time so every particle shares x_obs."
        )
    return keras.ops.repeat(resolved, num_samples, axis=0)
