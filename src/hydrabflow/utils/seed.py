"""Seeding helpers for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and (best effort) Keras.

    Returns a ``Generator`` to thread through simulators and preprocessing, so randomness is
    explicit rather than global.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # keras is optional at import time; only seed it if available
        import keras

        keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover - backend not installed in pure-config contexts
        pass
    return np.random.default_rng(seed)
