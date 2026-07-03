"""Helpers for compositional (grouped) evaluation.

A compositional dataset stores ``n`` groups of ``m`` exchangeable members (e.g. m streams
sharing one potential): globals ``(n, 1)``, per-member arrays ``(n, m, ...)``, group-level
observables (like the rotation curve) ``(n, bins, 1)``. Augmentations operate on flat rows, so
evaluation flattens the member axis, augments once, and regroups before sampling.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping

import numpy as np


def composition_level(cfg) -> str:
    return str(getattr(getattr(cfg, "composition", None), "level", "none") or "none")


def prior_score_from_spec(prior_spec: Mapping[str, Mapping]) -> Callable[[dict], dict]:
    """Score of the log prior for compositional sampling, from a prior-spec mapping.

    Returns a time-less callable (BayesFlow multiplies by ``(1 - t)`` itself): uniform priors
    contribute zero score, normal priors ``-(x - mean) / std**2``. The function is traced
    inside the sampler's integration loop, so it uses backend ops, not NumPy.
    """
    from keras import ops

    def score(x: Dict[str, np.ndarray], time=None) -> Dict[str, np.ndarray]:
        out = {}
        for key, arr in x.items():
            spec = prior_spec[key]
            if spec["type"] == "normal":
                mean, std = (float(p) for p in spec["prior_parameters"])
                out[key] = -(arr - mean) / std**2
            else:  # uniform (flat inside the support)
                out[key] = ops.zeros_like(arr)
        return out

    return score


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
