"""Preprocessing step protocol and the pipeline that orchestrates them.

A :class:`PreprocessStep` transforms a dataset dict (``{key: array}``) and may carry fitted state
(e.g. standardization mean/std). The :class:`PreprocessPipeline` runs an ordered list of steps:

* steps **before** the (optional) :class:`SplitStep` see the full dataset (e.g. NaN cleaning);
* the split divides data into train / validation;
* steps **after** the split are fit on the train split and applied to both splits.

At inference time ``transform`` replays the *fitted* element-wise steps (skipping the split), so
real / test data is processed identically to training. Fitted state round-trips through
``save`` / ``load`` (a single ``.npz`` in the run dir).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

Dataset = Dict[str, np.ndarray]


class PreprocessStep(ABC):
    """Element-wise (dataset-in, dataset-out) transform with optional fitted state."""

    name: str = "step"

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

    def inverse_transform(self, data: Dataset) -> Dataset:
        """Undo :meth:`transform` where meaningful (e.g. map normalized posterior samples back
        to physical units). Steps without a meaningful inverse return ``data`` unchanged."""
        return data


class SplitStep(PreprocessStep):
    """Marker base for the train/validation split (handled specially by the pipeline)."""

    def transform(self, data: Dataset) -> Dataset:  # pragma: no cover - never called directly
        return data

    @abstractmethod
    def split(self, data: Dataset, rng: np.random.Generator) -> Tuple[Dataset, Dataset]:
        """Return ``(train, val)``."""


class PreprocessPipeline:
    def __init__(self, steps: list[PreprocessStep]) -> None:
        self.steps = steps

    def fit_transform(
        self, data: Dataset, rng: np.random.Generator
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """Fit on the train split and transform train (+ val if a split is present)."""
        train: Dataset = data
        val: Optional[Dataset] = None
        for step in self.steps:
            if isinstance(step, SplitStep):
                train, val = step.split(train, rng)
                continue
            step.fit(train)
            train = step.transform(train)
            if val is not None:
                val = step.transform(val)
        return train, val

    def transform(self, data: Dataset) -> Dataset:
        """Inference path: apply fitted element-wise steps, skipping the split."""
        for step in self.steps:
            if isinstance(step, SplitStep):
                continue
            data = step.transform(data)
        return data

    def inverse_transform(self, data: Dataset) -> Dataset:
        """Undo the fitted element-wise steps in reverse order (posterior samples -> physical
        units). Steps without an inverse pass through unchanged."""
        for step in reversed(self.steps):
            if isinstance(step, SplitStep):
                continue
            data = step.inverse_transform(data)
        return data

    def get_step(self, name: str) -> Optional[PreprocessStep]:
        """First step registered under ``name`` (augmentations use this to read fitted state)."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    # ----------------------------------------------------------------------------------------- #
    # Persistence: one flat .npz. Keys are prefixed by step name + occurrence index (not list
    # position), so state fitted under the training pipeline still loads when an inference-time
    # pipeline variant (e.g. a real-data preset) arranges the same steps differently.
    # ----------------------------------------------------------------------------------------- #
    def _prefixes(self) -> list[str]:
        seen: Dict[str, int] = {}
        prefixes = []
        for step in self.steps:
            occurrence = seen.get(step.name, 0)
            seen[step.name] = occurrence + 1
            prefixes.append(f"{step.name}#{occurrence}.")
        return prefixes

    def save(self, path: str) -> None:
        flat: Dict[str, np.ndarray] = {}
        for prefix, step in zip(self._prefixes(), self.steps):
            for key, arr in step.state().items():
                flat[f"{prefix}{key}"] = arr
        np.savez(path, **flat)

    def load(self, path: str) -> None:
        raw = np.load(path, allow_pickle=True)
        for prefix, step in zip(self._prefixes(), self.steps):
            state = {k[len(prefix):]: raw[k] for k in raw.files if k.startswith(prefix)}
            if state:
                step.load_state(state)
