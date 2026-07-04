"""Built-in stateless preprocessing steps (besides standardization).

These mirror the deterministic dataset cleanup the reference project did inline before training
(NaN removal + train/val split, main_train_new_rotationcurve_agama.py:54-96), generalized and
made reusable / config-driven. Add your own by subclassing PreprocessStep and registering it.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from hydrabflow.preprocessing.base import Dataset, PreprocessStep, SplitStep
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
            x = np.asarray(data[key]).reshape(n, -1)
            valid &= np.isfinite(x).all(axis=1)
        if valid.all():
            return data
        return {k: np.asarray(v)[valid] for k, v in data.items()}


@register_step("cast_dtype")
class CastDtype(PreprocessStep):
    """Cast the listed keys (or all keys) to a target dtype, e.g. float32 for training."""

    name = "cast_dtype"

    def __init__(self, dtype: str = "float32", keys: Iterable[str] | None = None) -> None:
        self.dtype = dtype
        self.keys = list(keys) if keys is not None else None

    def transform(self, data: Dataset) -> Dataset:
        keys = self.keys if self.keys is not None else list(data.keys())
        out = dict(data)
        for key in keys:
            out[key] = np.asarray(data[key]).astype(self.dtype)
        return out


@register_step("select_keys")
class SelectKeys(PreprocessStep):
    """Keep only the listed keys (drop everything else)."""

    name = "select_keys"

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = list(keys)

    def transform(self, data: Dataset) -> Dataset:
        return {k: data[k] for k in self.keys if k in data}


@register_step("log10_transform")
class Log10Transform(PreprocessStep):
    """log10 the listed (strictly positive) keys in place; ``inverse_transform`` applies
    ``10**x``, so posterior samples come back in physical units. Reparametrizing a positive
    quantity this way also guarantees any real-valued network output maps back to a positive
    physical value — a cheap way to enforce positivity for parameters like a disk scale radius
    that a normal prior alone does not constrain to be positive.

    Downstream compositional-sampling prior scores must account for this change of variables
    (see ``pipeline.compositional.prior_score_from_spec``'s ``log10_keys`` argument) — the prior
    in ``conf/simulator/*.yaml`` is specified in physical units, not log10 space.
    """

    name = "log10_transform"

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = list(keys)

    def transform(self, data: Dataset) -> Dataset:
        out = dict(data)
        for key in self.keys:
            if key in out:
                out[key] = np.log10(np.asarray(out[key]))
        return out

    def inverse_transform(self, data: Dataset) -> Dataset:
        out = dict(data)
        for key in self.keys:
            if key in out:
                out[key] = 10.0 ** np.asarray(out[key])
        return out


@register_step("train_val_split")
class TrainValSplit(SplitStep):
    """Random hold-out split. Steps listed after this one are fit on the train split only."""

    name = "train_val_split"

    def __init__(self, validation_fraction: float = 0.1) -> None:
        self.validation_fraction = float(validation_fraction)

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
