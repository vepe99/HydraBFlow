"""Observational-noise augmentation — and the template for your own.

An augmentation is a factory ``(params, rng, context) -> (batch -> batch)``: it receives the shared
``augmentation.params`` mapping from config, its own seeded generator, and the run ``context`` (the
fitted preprocessing pipeline lives under ``"pipeline"``). All randomness must come from the
injected ``rng``, never from global ``np.random``: the generator advances on every call, so draws
differ batch to batch and epoch to epoch, yet the same ``cfg.seed`` reproduces the exact sequence.

Register with ``@register_augmentation("name")``, list the name under ``augmentation.steps``, and
put its knobs under ``augmentation.params``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from hydrabflow.augmentation.registry import Augmentation, register_augmentation


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
