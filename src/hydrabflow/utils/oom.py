"""Retry a GPU computation with a progressively smaller batch size when it runs out of memory.

BayesFlow/JAX raise a ``RESOURCE_EXHAUSTED`` (``jax.errors.JaxRuntimeError``) when a batch is too
large for the card — during ``fit_offline`` (training) or ``sample`` (posterior sampling, where the
flattened ``batch_size * num_samples`` tensor can be large). Rather than fail the whole run/trial,
:func:`run_with_oom_backoff` catches the OOM, halves the batch size, clears the compilation cache,
and retries — down to ``min_batch``. Non-OOM errors propagate unchanged.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


def is_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a GPU out-of-memory error (backend-agnostic, matches on message)."""
    text = f"{type(exc).__name__}: {exc}"
    needles = ("RESOURCE_EXHAUSTED", "out of memory", "OOM", "CUDA_ERROR_OUT_OF_MEMORY")
    return any(n.lower() in text.lower() for n in needles)


def run_with_oom_backoff(
    fn: Callable[[int], T],
    batch_size: int,
    *,
    min_batch: int = 16,
    factor: int = 2,
    logger=None,
) -> T:
    """Call ``fn(batch_size)``; on an OOM error halve the batch size and retry until ``min_batch``.

    ``fn`` must accept a single ``batch_size`` argument and re-run the full computation with it
    (idempotent w.r.t. earlier failed attempts). The last OOM is re-raised if even ``min_batch``
    cannot fit. Non-OOM exceptions are never swallowed.
    """
    bs = int(batch_size)
    min_batch = max(1, int(min_batch))
    while True:
        try:
            return fn(bs)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM we can back off from
            if not is_oom_error(exc) or bs <= min_batch:
                raise
            new_bs = max(min_batch, bs // int(factor))
            if new_bs >= bs:  # cannot shrink further
                raise
            if logger is not None:
                logger.warning(
                    "GPU OOM at batch_size=%d; retrying at batch_size=%d (min %d).",
                    bs, new_bs, min_batch,
                )
            bs = new_bs
            # Release the failed allocation + compiled executables before retrying.
            try:
                import gc

                gc.collect()
                import jax

                jax.clear_caches()
            except Exception:  # noqa: BLE001 - best-effort cleanup; never mask the retry
                pass
