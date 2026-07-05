"""Helpers for compositional (grouped) evaluation.

A compositional dataset stores ``n`` groups of ``m`` exchangeable members (e.g. m streams
sharing one potential): globals ``(n, 1)``, per-member arrays ``(n, m, ...)``, group-level
observables (like the rotation curve) ``(n, bins, 1)``. Augmentations operate on flat rows, so
evaluation flattens the member axis, augments once, and regroups before sampling.
"""

from __future__ import annotations

import math
from typing import Callable, Container, Dict, Mapping

import numpy as np


def composition_level(cfg) -> str:
    return str(getattr(getattr(cfg, "composition", None), "level", "none") or "none")


def prior_score_from_spec(
    prior_spec: Mapping[str, Mapping], log10_keys: Container[str] = ()
) -> Callable[[dict], dict]:
    """Score of the log prior for compositional sampling, from a prior-spec mapping.

    BayesFlow decides whether to apply its own ``(1 - t)`` time-decay by inspecting this
    callable's signature (``bayesflow.approximators.helpers.compositional``:
    ``prior_has_time = "time" in inspect.signature(compute_prior_score).parameters``): if the
    callable accepts a ``time`` argument, BayesFlow assumes the callable handles the decay
    itself and skips its own multiplication. Since this function's signature names a ``time``
    parameter, it MUST apply the ``(1 - t)`` factor explicitly here (matching the compositional
    formula ``(1-n)(1-t) grad log p(theta) + sum_i s_i``) — otherwise the raw, undecayed prior
    score is injected at every diffusion step (including near t=1, pure noise), which showed up
    as severe under-confidence for tight normal priors (e.g. z_Disk) while uniform priors were
    unaffected (their raw score is 0 regardless of decay).

    Uniform priors contribute zero score, normal priors ``-(x - mean) / std**2``. The function
    is traced inside the sampler's integration loop, so it uses backend ops, not NumPy.

    ``log10_keys`` lists parameters the ``log10_transform`` preprocessing step reparametrized
    (see ``preprocessing.steps.Log10Transform``): the prior in ``prior_spec`` (from the
    simulator's config) is defined in physical units ``x``, but the network — and hence this
    score, evaluated during compositional sampling — operates on ``y = log10(x)``. By the
    change-of-variables rule, for ``x = g(y) = 10**y`` (so ``g'(y) = g(y) * ln(10)``):

        d/dy log p_Y(y) = g'(y) * [d/dx log p_X(x)]|_{x=g(y)} + d/dy log(g'(y))
                         = x * ln(10) * score_X(x) + ln(10)

    (the ``+ln(10)`` term is the Jacobian's own contribution and is nonzero even for a uniform
    prior in ``x`` — omitting it silently biases the compositional posterior.)
    """
    from keras import ops

    ln10 = math.log(10.0)

    def score(x: Dict[str, np.ndarray], time=None) -> Dict[str, np.ndarray]:
        out = {}
        for key, arr in x.items():
            spec = prior_spec[key]
            is_log10 = key in log10_keys
            x_phys = 10.0**arr if is_log10 else arr

            if spec["type"] == "normal":
                mean, std = (float(p) for p in spec["prior_parameters"])
                base_score = -(x_phys - mean) / std**2
            else:  # uniform (flat inside the support)
                base_score = ops.zeros_like(arr)

            if is_log10:
                score_val = x_phys * ln10 * base_score + ln10
            else:
                score_val = base_score

            out[key] = score_val if time is None else (1.0 - time) * score_val
        return out

    return score


def log10_keys_from_pipeline(pipeline) -> list:
    """Keys the ``log10_transform`` preprocessing step reparametrized, if present (else empty)."""
    step = pipeline.get_step("log10_transform") if pipeline is not None else None
    return list(step.keys) if step is not None else []


def flatten_members(data: Mapping[str, np.ndarray], m: int) -> Dict[str, np.ndarray]:
    """Reshape per-member arrays ``(n, m, ...) -> (n*m, ...)``; repeat group-level arrays
    (globals, rotation curve) ``m`` times so every flat row is complete.

    An array is treated as per-member when its second axis has length ``m`` and it has at least
    three dimensions — group-level observables must therefore not have ``m`` bins on axis 1.
    """
    out = {}
    for key, arr in data.items():
        arr = np.asarray(arr)
        if arr.ndim >= 3 and arr.shape[1] == m:
            out[key] = arr.reshape(arr.shape[0] * m, *arr.shape[2:])
        else:
            out[key] = np.repeat(arr, m, axis=0)
    return out


def group_members(data: Mapping[str, np.ndarray], n: int, m: int) -> Dict[str, np.ndarray]:
    """Inverse of :func:`flatten_members` for arrays of ``n*m`` rows -> ``(n, m, ...)``."""
    return {
        key: np.asarray(arr).reshape(n, m, *np.asarray(arr).shape[1:])
        for key, arr in data.items()
    }


def condition_keys(cfg) -> list:
    """Raw batch keys that act as sampling conditions (everything the adapter consumes except
    the inference targets)."""
    from hydrabflow.pipeline.adapter import adapter_keys

    targets = set(cfg.adapter.inference_variables)
    return [k for k in adapter_keys(cfg) if k not in targets]


def apply_augmentations_once(flat: Dict[str, np.ndarray], cfg, pipeline, seed: int):
    """Replay the configured augmentation chain once (fixed draw) on flattened rows."""
    from hydrabflow.augmentation.registry import build_augmentations

    augmentations = build_augmentations(
        cfg.augmentation, np.random.default_rng(seed), context={"pipeline": pipeline}
    )
    for aug in augmentations:
        flat = aug(flat)
    return {k: np.asarray(v) for k, v in flat.items()}
