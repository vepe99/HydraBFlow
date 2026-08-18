"""Observational-noise augmentations — and the template for your own.

An augmentation is a factory ``(params, rng, context) -> (batch -> batch)``, given the shared
``augmentation.params``, its own seeded generator, and the run ``context`` (the fitted preprocessing
pipeline is under ``"pipeline"``). Draw only from the injected ``rng``, never global ``np.random``,
so the same ``cfg.seed`` reproduces the whole sequence.

To add one: ``@register_augmentation("name")``, list the name under ``augmentation.steps``, knobs
under ``augmentation.params``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import jax
from jax import random
import jax.numpy as jnp

from hydrabflow.registry import Augmentation, register_augmentation


@register_augmentation("gaussian_noise")
def gaussian_noise(params: Mapping, rng: np.random.Generator, context: dict) -> Augmentation:
    """Add zero-mean Gaussian noise to one observable key.

    Params: ``noise_key`` (key to perturb, default ``"x"``) and ``noise_scale`` (std, default
    ``0.0`` — a no-op, so the shipped config trains without augmentation).
    """
    key = params.get("noise_key", "x")
    scale = float(params.get("noise_scale", 0.0))

    def _apply(batch: dict) -> dict:
        if scale > 0.0 and key in batch:
            x = np.asarray(batch[key])
            batch[key] = x + rng.normal(0.0, scale, size=x.shape).astype(x.dtype)
        return batch

    return _apply

@register_augmentation("jax_noise")
def jax_noise(params, rng, context):
    key_root = random.key(int(rng.integers(2**63)))   # derive from the numpy child -> still cfg.seed
    scale = float(params.get("noise_scale", 0.1))
    step = {"n": 0}                                    # explicit call counter

    @jax.jit
    def _kernel(k, x):
        return x + scale * random.normal(k, x.shape, x.dtype)

    def _apply(batch):
        k = random.fold_in(key_root, step["n"])
        step["n"] += 1
        batch["x"] = _kernel(k, jnp.asarray(batch["x"]))
        return batch

    return _apply
