"""Retry a GPU computation at a smaller batch size when it runs out of memory.

JAX raises ``RESOURCE_EXHAUSTED`` when a batch is too large for the card. Rather than fail the run,
halve the batch and retry down to ``min_batch``, clearing the compilation cache each time. Non-OOM
errors propagate unchanged.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


def is_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a GPU out-of-memory error (matches on the message)."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        n in text for n in ("resource_exhausted", "out of memory", "oom", "cuda_error_out_of_memory")
    )


def run_with_oom_backoff(
    fn: Callable[[int], T],
    batch_size: int,
    *,
    min_batch: int = 16,
    logger=None,
) -> T:
    """Call ``fn(batch_size)``, halving the batch size on OOM until ``min_batch``.

    ``fn`` must re-run the full computation for the batch size it is given. The last OOM is
    re-raised if even ``min_batch`` cannot fit.
    """
    bs = int(batch_size)
    min_batch = max(1, int(min_batch))
    while True:
        try:
            return fn(bs)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM we can back off from
            if not is_oom_error(exc) or bs <= min_batch:
                raise
            new_bs = max(min_batch, bs // 2)
            if logger is not None:
                logger.warning("GPU OOM at batch_size=%d; retrying at %d (min %d).",
                               bs, new_bs, min_batch)
            bs = new_bs
            # Release the failed allocation + compiled executables before retrying.
            try:
                import gc

                import jax

                gc.collect()
                jax.clear_caches()
            except Exception:  # noqa: BLE001 - best-effort cleanup; never mask the retry
                pass
