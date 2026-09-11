"""Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose leading axis
is the number of simulations (one (parameters, observation) pair per row).
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Callable, Dict

import numpy as np
from tqdm import tqdm

from hydrabflow.utils.progress import set_row_tick

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
    data = {k: raw[k] for k in raw.files}
    log.info("Loaded dataset (%d rows, keys=%s) <- %s", n_rows(data), list(data), path)
    return data


def _save_chunk_atomic(path: str, data: Dataset) -> None:
    """Write a chunk atomically (temp file + rename) so a crash mid-write can never leave a
    truncated chunk that a later resume would mistake for valid."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:  # a file object, so numpy does not append a second ``.npz``
        np.savez(f, **data)
    os.replace(tmp, path)


def run_chunked(
    out_path: str,
    n_total: int,
    chunk: int,
    sample_fn: Callable[[int, np.random.Generator], Dataset],
    base_seed: int,
    desc: str = "simulating",
) -> str:
    """Generate ``n_total`` rows in chunks of ``chunk``, checkpointing each chunk to disk, and
    assemble them into the final dataset at ``out_path``.

    Each chunk gets its own RNG seeded from ``(base_seed, row_offset)``, so it is reproducible
    independently of the others. The offsets themselves depend on ``chunk``, so changing the chunk
    size changes the draws (same distribution, different sample) — it is a memory knob, not a free
    parameter of a fixed dataset.

    Resumable: chunks land in a sidecar ``<stem>.chunks/`` directory; on restart any valid chunk
    already present is loaded instead of re-simulated, so a crash only costs the in-flight chunk
    (the expensive stream simulators run for hours). On success the sidecar directory is removed.
    """
    stem = out_path[:-4] if out_path.endswith(".npz") else out_path
    chunk_dir = stem + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    # One row-granular bar for the whole run. A joblib simulator advances it per finished row via
    # utils.progress; simulators with no per-row hook leave it untouched and we step it per chunk.
    starts = list(range(0, n_total, chunk))
    bar = tqdm(total=n_total, desc=desc, unit="row", dynamic_ncols=True, smoothing=0.02)
    set_row_tick(bar.update)
    try:
        for idx, start in enumerate(starts):
            n = min(chunk, n_total - start)
            cpath = os.path.join(chunk_dir, f"chunk_{idx:05d}.npz")
            if os.path.exists(cpath):
                try:
                    if n_rows(load_chunk(cpath)) == n:
                        log.info("resume: chunk %d/%d present, skipping", idx + 1, len(starts))
                        bar.update(n)
                        continue
                    log.warning("chunk %d has the wrong row count, regenerating", idx)
                except Exception as exc:  # corrupt/partial file -> regenerate
                    log.warning("chunk %d unreadable (%s), regenerating", idx, exc)
            rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), int(start)]))
            before = bar.n
            _save_chunk_atomic(cpath, sample_fn(n, rng))
            if bar.n == before:  # simulator did not report per-row progress -> step per chunk
                bar.update(n)
    finally:
        set_row_tick(None)
        bar.close()

    chunks = [load_chunk(os.path.join(chunk_dir, f"chunk_{i:05d}.npz")) for i in range(len(starts))]
    keys = chunks[0].keys() if chunks else []
    save_dataset(out_path, {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys})
    shutil.rmtree(chunk_dir, ignore_errors=True)
    return out_path


def load_chunk(path: str) -> Dataset:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}
