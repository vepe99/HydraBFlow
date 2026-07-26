"""Natural-log transform for strictly positive observables.

Motivating case: the Lotka-Volterra prior predictive spans ~9 orders of magnitude (population
crashes to the extinction floor in some draws, explodes to 1e3 in others). Standardizing raw counts
is hopeless — the mean/std are dominated by a handful of exploding trajectories. Because the LV
observation noise is *multiplicative lognormal*, ``log x`` is also the scale on which the noise is
additive and homoscedastic, so this is the natural representation for the summary network.

Place it **before** ``standardize`` in the pipeline (log first, then centre/scale).

Config (``conf/preprocessing/lv.yaml``)::

    - name: log_transform
      keys: ${adapter.summary_variables}
      floor: 1.0e-6

``floor`` clips the input from below before taking the log, so an exact-zero or clamped-extinction
entry becomes a large-but-finite negative number instead of ``-inf``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from hydrabflow.preprocessing.base import Dataset, PreprocessStep
from hydrabflow.preprocessing.registry import register_step


@register_step("log_transform")
class LogTransform(PreprocessStep):
    """Stateless ``x -> log(max(x, floor))``, inverted by ``exp``."""

    name = "log_transform"

    def __init__(self, keys: Iterable[str], floor: float = 1e-6) -> None:
        self.keys = list(keys)
        self.floor = float(floor)
        if self.floor <= 0.0:
            raise ValueError(f"log_transform.floor must be positive, got {self.floor}")

    def transform(self, data: Dataset) -> Dataset:
        out = dict(data)
        for key in self.keys:
            if key not in out:
                continue
            out[key] = np.log(np.maximum(np.asarray(out[key], dtype=np.float64), self.floor))
        return out

    def inverse_transform(self, data: Dataset) -> Dataset:
        out = dict(data)
        for key in self.keys:
            if key not in out:
                continue
            out[key] = np.exp(np.asarray(out[key], dtype=np.float64))
        return out
