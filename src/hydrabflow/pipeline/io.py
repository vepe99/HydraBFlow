"""Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose leading axis
is the number of simulations (one (parameters, observation) pair per row).
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)
Dataset = Dict[str, np.ndarray]


def n_rows(data: Dataset) -> int:
    """Number of simulations in a dataset dict (length of its leading axis)."""
    return len(next(iter(data.values()))) if data else 0


def save_dataset(path: str, data: Dataset) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **data)
    log.info("Saved dataset (%d rows, keys=%s) -> %s", n_rows(data), list(data), path)


def load_dataset(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}. Generate it first (e.g. `hydrabflow-simulate`)."
        )
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def run_chunked(
    out_path: str,
    n_total: int,
    chunk: int,
    sample_fn: Callable[[int, np.random.Generator], Dataset],
    base_seed: int,
) -> str:
    """Generate ``n_total`` rows in chunks of ``chunk`` and save them to ``out_path``.

    Each chunk gets its own RNG seeded from ``(base_seed, row_offset)``, so it is reproducible
    independently of the others and the chunk size never changes the dataset.
    """
    chunks = []
    for start in tqdm(range(0, n_total, chunk), desc="simulating", unit="chunk"):
        rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), int(start)]))
        chunks.append(sample_fn(min(chunk, n_total - start), rng))

    keys = chunks[0].keys() if chunks else []
    save_dataset(out_path, {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys})
    return out_path
