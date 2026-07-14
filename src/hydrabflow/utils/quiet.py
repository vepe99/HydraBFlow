"""Silence noisy C-extension output (AGAMA) during simulation.

AGAMA writes diagnostics straight to file descriptors 1/2 at the C level, so a Python-level
``contextlib.redirect_stdout`` cannot catch them — the redirect must happen at the OS fd level
(``os.dup2``). Simulation runs this inside the joblib workers, each of which is its own process,
so redirecting their fds is isolated and never touches the parent's terminal or joblib's IPC
(joblib communicates over its own pipes, not fd 1/2).

Gated by ``HYDRABFLOW_SIM_QUIET`` (default on). Set ``HYDRABFLOW_SIM_QUIET=0`` to see AGAMA's
output again when debugging a simulator.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys


def sim_quiet_enabled() -> bool:
    """True unless ``HYDRABFLOW_SIM_QUIET`` is set to a falsy value."""
    return os.environ.get("HYDRABFLOW_SIM_QUIET", "1").lower() not in ("0", "false", "no", "")


@contextlib.contextmanager
def suppress_c_stdio():
    """Redirect fd 1 & 2 to ``/dev/null`` for the duration of the block (C-level output included).

    No-op when quiet mode is disabled. Restores the original fds on exit even if the body raises.
    """
    if not sim_quiet_enabled():
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


def quiet_worker(fn):
    """Decorator: run a (joblib) worker with its C-level stdout/stderr redirected to /dev/null."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with suppress_c_stdio():
            return fn(*args, **kwargs)

    return wrapper
