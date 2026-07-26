"""Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose leading axis
is the number of simulations (one (parameters, observation) pair per row).
"""

from __future__ import annotations

import os
import shutil
from typing import Callable, Dict

import numpy as np
from tqdm import tqdm

from hydrabflow.utils.logging import get_logger
from hydrabflow.utils.progress import set_row_tick

log = get_logger(__name__)
Dataset = Dict[str, np.ndarray]


def _n_rows(data: Dataset) -> int:
    return len(next(iter(data.values()))) if data else 0


def save_dataset(path: str, data: Dataset) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **data)
    n = len(next(iter(data.values()))) if data else 0
    log.info("Saved dataset (%d rows, keys=%s) -> %s", n, list(data), path)


def load_dataset(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}. Generate it first (e.g. `hydrabflow-simulate`)."
        )
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    n = len(next(iter(data.values()))) if data else 0
    log.info("Loaded dataset (%d rows, keys=%s) <- %s", n, list(data), path)
    return data


def concatenate_chunks(chunks: list[Dataset]) -> Dataset:
    """Concatenate a list of dataset dicts along the leading (simulation) axis."""
    if not chunks:
        return {}
    keys = chunks[0].keys()
    return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}


def _save_chunk_atomic(path: str, data: Dataset) -> None:
    """Write a chunk ``.npz`` atomically (temp file + rename), so a crash mid-write can never
    leave a truncated chunk that a later resume would mistake for valid."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    # Pass a file object (not a name) so numpy does not append a second ``.npz`` extension.
    with open(tmp, "wb") as f:
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
    """Generate ``n_total`` rows in chunks, checkpointing each chunk to disk as it completes,
    and assemble them into the final dataset at ``out_path``.

    Resumable: chunks land in a sidecar ``<out_path stem>.chunks/`` directory; on restart any
    valid chunk already present is loaded instead of re-simulated, so a crash only costs the
    in-flight chunk. Each chunk gets its own RNG seeded from ``(base_seed, row_offset)``, so a
    chunk is reproducible independently of the others and a resumed run yields the exact same
    dataset a single uninterrupted run would have (given the same ``chunk`` size). On success the
    sidecar directory is removed; leave a failed run's directory in place to resume from it.
    """
    stem = out_path[:-4] if out_path.endswith(".npz") else out_path
    chunk_dir = stem + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    # One row-granular bar for the whole run. A joblib simulator advances it per finished row via
    # utils.progress (published below); simulators with no per-row hook leave it untouched and we
    # step it per chunk instead. This is the only thing printed during generation — AGAMA's own
    # chatter is redirected to /dev/null in the workers (utils.quiet).
    starts = list(range(0, n_total, chunk))
    bar = tqdm(total=n_total, desc=desc, unit="row", dynamic_ncols=True, smoothing=0.02)
    set_row_tick(bar.update)
    try:
        for idx, start in enumerate(starts):
            n = min(chunk, n_total - start)
            cpath = os.path.join(chunk_dir, f"chunk_{idx:05d}.npz")
            if os.path.exists(cpath):
                try:
                    existing = load_chunk(cpath)
                    if _n_rows(existing) == n:
                        log.info("resume: chunk %d/%d present (%d rows), skipping", idx + 1, len(starts), n)
                        bar.update(n)
                        continue
                    log.warning(
                        "chunk %d row mismatch (%d != %d), regenerating", idx, _n_rows(existing), n
                    )
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

    chunks = [
        load_chunk(os.path.join(chunk_dir, f"chunk_{idx:05d}.npz")) for idx in range(len(starts))
    ]
    data = concatenate_chunks(chunks)
    save_dataset(out_path, data)
    shutil.rmtree(chunk_dir, ignore_errors=True)
    return out_path


def load_chunk(path: str) -> Dataset:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}
