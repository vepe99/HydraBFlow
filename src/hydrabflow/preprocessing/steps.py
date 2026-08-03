"""Built-in stateless preprocessing steps (besides standardization).

Add your own by subclassing PreprocessStep, decorating it with ``@register_step("name")``, and
listing that name under ``preprocessing.steps`` in ``conf/config.yaml``.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from hydrabflow.preprocessing.base import Dataset, PreprocessStep
from hydrabflow.preprocessing.registry import register_step


@register_step("drop_nan")
class DropNaNSimulations(PreprocessStep):
    """Drop rows (simulations) that contain any NaN/Inf in the listed keys."""

    name = "drop_nan"

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = list(keys)

    def transform(self, data: Dataset) -> Dataset:
        n = _num_rows(data)
        valid = np.ones(n, dtype=bool)
        for key in self.keys:
            valid &= np.isfinite(np.asarray(data[key]).reshape(n, -1)).all(axis=1)
        if valid.all():
            return data
        return {k: np.asarray(v)[valid] for k, v in data.items()}


@register_step("train_val_split")
class TrainValSplit(PreprocessStep):
    """Random hold-out split. Steps listed after this one are fit on the train split only."""

    name = "train_val_split"
    splits = True

    def __init__(self, validation_fraction: float = 0.1) -> None:
        self.validation_fraction = float(validation_fraction)

    def transform(self, data: Dataset) -> Dataset:  # pragma: no cover - the pipeline calls split()
        return data

    def split(self, data: Dataset, rng: np.random.Generator) -> Tuple[Dataset, Dataset]:
        n = _num_rows(data)
        n_val = int(round(n * self.validation_fraction))
        perm = rng.permutation(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train = {k: np.asarray(v)[train_idx] for k, v in data.items()}
        val = {k: np.asarray(v)[val_idx] for k, v in data.items()}
        return train, val


def _num_rows(data: Dataset) -> int:
    return len(next(iter(data.values())))
