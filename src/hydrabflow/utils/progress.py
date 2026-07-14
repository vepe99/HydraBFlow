"""A single, live, row-granular progress bar for dataset generation.

The dataset generator (``pipeline.io.run_chunked``) owns one ``tqdm`` bar spanning the whole run
and publishes its ``update`` here via :func:`set_row_tick`. A joblib-based simulator can then
advance that same bar once per finished row — without knowing anything about the bar — by running
its ``Parallel(...)`` call inside :func:`joblib_row_progress`. Simulators with no per-row hook
(e.g. ``two_moons``) simply don't tick, and ``run_chunked`` advances the bar per chunk instead.

Kept in ``utils`` (no pipeline import) so simulators can use it without a circular dependency.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Optional

_row_tick: Optional[Callable[[int], None]] = None


def set_row_tick(fn: Optional[Callable[[int], None]]) -> None:
    """Register (or clear, with ``None``) the callback that advances the active progress bar."""
    global _row_tick
    _row_tick = fn


def row_tick(n: int = 1) -> None:
    """Advance the active progress bar by ``n`` rows, if one is registered."""
    if _row_tick is not None:
        try:
            _row_tick(n)
        except Exception:  # a progress bar must never break a simulation
            pass


@contextlib.contextmanager
def joblib_row_progress():
    """Advance the registered row tick as joblib batches complete (one tick per finished task).

    No-op when no tick is registered (so the simulator behaves normally outside ``run_chunked``,
    e.g. in tests or tuning). Patches joblib's batch-completion callback for the duration of the
    block and restores it afterwards.
    """
    if _row_tick is None:
        yield
        return
    import joblib.parallel as jp

    original = jp.BatchCompletionCallBack

    class _TickingCallBack(original):  # type: ignore[misc, valid-type]
        def __call__(self, *args, **kwargs):
            row_tick(int(getattr(self, "batch_size", 1)))
            return super().__call__(*args, **kwargs)

    jp.BatchCompletionCallBack = _TickingCallBack
    try:
        yield
    finally:
        jp.BatchCompletionCallBack = original
