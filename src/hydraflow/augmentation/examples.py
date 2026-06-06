"""Example augmentations. Use as templates for problem-specific ones.

Augmentations run per batch during training and should be cheap and stochastic. Each is built by
a factory ``(params, rng) -> (batch -> batch)``: it receives the shared ``augmentation.params``
mapping plus its own seeded :class:`~numpy.random.Generator`, and returns a callable that maps a
batch dict (of arrays) to a batch dict.

**Reproducibility contract** (see ``augmentation/registry.py``): all randomness must come from the
injected ``rng`` — never from the global ``np.random`` functions. The generator is stateful, so it
advances on every call: draws differ from batch to batch and epoch to epoch (genuinely
stochastic), yet rerunning with the same ``cfg.seed`` reproduces the exact same sequence. This
mirrors the reference project's ``AugmentationsClass._split_key`` pattern.

Register with ``@register_augmentation("name")`` and list the name under ``augmentation.steps``;
put any knobs under ``augmentation.params``. Every example below is a no-op at its default
strength (scale / probability == 0), matching the template's "trivial but valid" defaults.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from hydraflow.augmentation.registry import Augmentation, register_augmentation


@register_augmentation("gaussian_noise")
def gaussian_noise(params: Mapping, rng: np.random.Generator) -> Augmentation:
    """Add zero-mean Gaussian noise to one observable key (additive observational noise).

    Config params: ``noise_key`` (key to perturb, default ``"x"``) and ``noise_scale``
    (std, default ``0.0``).
    """
    key = params.get("noise_key", "x")
    scale = float(params.get("noise_scale", 0.0))

    def _apply(batch: dict) -> dict:
        if scale > 0.0 and key in batch:
            x = np.asarray(batch[key])
            batch[key] = x + rng.normal(0.0, scale, size=x.shape).astype(x.dtype)
        return batch

    return _apply


@register_augmentation("multiplicative_noise")
def multiplicative_noise(params: Mapping, rng: np.random.Generator) -> Augmentation:
    """Scale an observable by ``(1 + N(0, mult_scale))`` — multiplicative / gain jitter.

    Config params: ``noise_key`` (default ``"x"``) and ``mult_scale`` (relative std, default
    ``0.0``).
    """
    key = params.get("noise_key", "x")
    scale = float(params.get("mult_scale", 0.0))

    def _apply(batch: dict) -> dict:
        if scale > 0.0 and key in batch:
            x = np.asarray(batch[key])
            factor = 1.0 + rng.normal(0.0, scale, size=x.shape)
            batch[key] = (x * factor).astype(x.dtype)
        return batch

    return _apply


@register_augmentation("feature_dropout")
def feature_dropout(params: Mapping, rng: np.random.Generator) -> Augmentation:
    """Randomly zero out entries of an observable with probability ``dropout_prob`` (Bernoulli mask).

    A simple robustness augmentation (missing/corrupted measurements). Config params: ``noise_key``
    (default ``"x"``) and ``dropout_prob`` (per-entry drop probability in ``[0, 1)``, default
    ``0.0``).
    """
    key = params.get("noise_key", "x")
    prob = float(params.get("dropout_prob", 0.0))

    def _apply(batch: dict) -> dict:
        if prob > 0.0 and key in batch:
            x = np.asarray(batch[key])
            keep = (rng.random(size=x.shape) >= prob).astype(x.dtype)
            batch[key] = x * keep
        return batch

    return _apply
