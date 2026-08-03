"""Preprocessing step protocol and the pipeline that orchestrates them.

A :class:`PreprocessStep` transforms a dataset dict (``{key: array}``) and may carry fitted state
(e.g. standardization mean/std). The :class:`PreprocessPipeline` runs an ordered list of steps:

* steps **before** the splitting step see the full dataset (e.g. NaN cleaning);
* the splitting step divides data into train / validation;
* steps **after** it are fit on the train split and applied to both splits.

At inference time ``transform`` replays the *fitted* steps (skipping the split), so real / test data
is processed identically to training. Fitted state round-trips through ``save`` / ``load``
(a single ``.npz`` in the run dir).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

Dataset = Dict[str, np.ndarray]


class PreprocessStep(ABC):
    """Dataset-in, dataset-out transform with optional fitted state."""

    name: str = "step"
    #: True for the train/val splitting step, which the pipeline handles specially: it implements
    #: ``split(data, rng) -> (train, val)`` instead of ``transform`` (see ``steps.TrainValSplit``).
    splits: bool = False

    def fit(self, data: Dataset) -> None:  # noqa: B027 - intentional no-op default
        """Estimate any state from ``data`` (train split). Stateless steps leave this empty."""

    @abstractmethod
    def transform(self, data: Dataset) -> Dataset:
        """Return a transformed copy/view of ``data``."""

    def state(self) -> Dict[str, np.ndarray]:
        """Arrays to persist so the fitted transform can be reloaded. Default: nothing."""
        return {}

    def load_state(self, state: Dict[str, np.ndarray]) -> None:  # noqa: B027
        """Restore arrays produced by :meth:`state`."""


class PreprocessPipeline:
    def __init__(self, steps: list[PreprocessStep]) -> None:
        self.steps = steps

    def fit_transform(
        self, data: Dataset, rng: np.random.Generator
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """Fit on the train split and transform train (+ val if a splitting step is present)."""
        train: Dataset = data
        val: Optional[Dataset] = None
        for step in self.steps:
            if step.splits:
                train, val = step.split(train, rng)
                continue
            step.fit(train)
            train = step.transform(train)
            if val is not None:
                val = step.transform(val)
        return train, val

    def transform(self, data: Dataset) -> Dataset:
        """Inference path: apply the fitted steps, skipping the split."""
        for step in self.steps:
            if not step.splits:
                data = step.transform(data)
        return data

    # Persistence: one flat .npz, keys prefixed by step name.
    def save(self, path: str) -> None:
        flat = {
            f"{step.name}.{key}": arr for step in self.steps for key, arr in step.state().items()
        }
        np.savez(path, **flat)

    def load(self, path: str) -> None:
        raw = np.load(path, allow_pickle=True)
        for step in self.steps:
            prefix = f"{step.name}."
            state = {k[len(prefix):]: raw[k] for k in raw.files if k.startswith(prefix)}
            if state:
                step.load_state(state)
