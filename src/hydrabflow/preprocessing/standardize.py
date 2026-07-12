"""Per-feature z-score standardization step.

Generalizes the reference project's ``compute_standardization`` / ``apply_standardization`` /
``save_stats`` / ``load_stats`` (utils_train_jax_new_rotationcurve.py:752-809): mean/std are fit
on the train split over all axes except the last (feature) axis, then reused everywhere.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from hydrabflow.preprocessing.base import Dataset, PreprocessStep
from hydrabflow.preprocessing.registry import register_step

_EPS = 1e-8


@register_step("standardize")
class Standardizer(PreprocessStep):
    name = "standardize"

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = list(keys)
        self._mean: Dict[str, np.ndarray] = {}
        self._std: Dict[str, np.ndarray] = {}

    def fit(self, data: Dataset) -> None:
        for key in self.keys:
            x = np.asarray(data[key])
            reduce_axes = tuple(range(x.ndim - 1))  # keep last (feature) axis
            self._mean[key] = x.mean(axis=reduce_axes)
            self._std[key] = x.std(axis=reduce_axes).clip(min=_EPS)

    def transform(self, data: Dataset) -> Dataset:
        out = dict(data)
        for key in self.keys:
            if key not in data:  # e.g. a condition/observable absent from this dict — skip (cf. Log10Transform)
                continue
            if key not in self._mean:
                raise RuntimeError(f"Standardizer not fitted for key '{key}'")
            out[key] = (np.asarray(data[key]) - self._mean[key]) / self._std[key]
        return out

    def inverse_transform(self, data: Dataset) -> Dataset:
        out = dict(data)
        for key in self.keys:
            if key not in data:  # posterior-sample dicts hold only inference_variables — skip conditions
                continue
            out[key] = np.asarray(data[key]) * self._std[key] + self._mean[key]
        return out

    def state(self) -> Dict[str, np.ndarray]:
        flat: Dict[str, np.ndarray] = {}
        for key in self.keys:
            flat[f"{key}__mean"] = self._mean[key]
            flat[f"{key}__std"] = self._std[key]
        return flat

    def load_state(self, state: Dict[str, np.ndarray]) -> None:
        for key in self.keys:
            self._mean[key] = state[f"{key}__mean"]
            self._std[key] = state[f"{key}__std"]
