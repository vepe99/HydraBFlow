"""Pin compute settings *before* keras/bayesflow/JAX are imported anywhere.

Both libraries read their configuration at import time, so GPU selection and the Keras backend must
be set first — which is why ``hydrabflow/__init__.py`` imports this module before anything else.

Environment overrides (this runs before Hydra config exists):
``CUDA_VISIBLE_DEVICES`` (set it yourself and autocvd is skipped), ``HYDRABFLOW_NUM_GPUS`` (how many
GPUs autocvd should expose, default 1; ``0`` forces CPU-only), ``KERAS_BACKEND`` (default ``jax``).
"""

from __future__ import annotations

import logging
import os

DEFAULT_BACKEND = "jax"
NUM_GPUS_ENV = "HYDRABFLOW_NUM_GPUS"
DEFAULT_NUM_GPUS = 1

_log = logging.getLogger(__name__)


def limit_gpus(num_gpus: int | None = None) -> str | None:
    """Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initializes.

    Returns the resulting value, or ``None`` if it was left unset (autocvd missing, or no NVIDIA
    GPUs / no ``nvidia-smi`` — all degrade to a log line rather than an error).
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]  # never override an explicit choice

    if num_gpus is None:
        try:
            num_gpus = int(os.environ.get(NUM_GPUS_ENV, DEFAULT_NUM_GPUS))
        except ValueError:
            _log.warning("Invalid %s=%r; falling back to %d.", NUM_GPUS_ENV,
                         os.environ.get(NUM_GPUS_ENV), DEFAULT_NUM_GPUS)
            num_gpus = DEFAULT_NUM_GPUS

    if num_gpus <= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only: hide all GPUs from JAX
        return ""

    try:
        from autocvd import autocvd

        autocvd(num_gpus=num_gpus)  # sets CUDA_VISIBLE_DEVICES to the least-utilized GPUs
    except Exception as exc:  # noqa: BLE001 — not installed, or no GPUs/nvidia-smi (macOS, CPU box)
        _log.warning("Could not select GPUs (%s); leaving CUDA_VISIBLE_DEVICES unset.", exc)
        return None
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def set_backend(backend: str = DEFAULT_BACKEND) -> str:
    """Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active backend."""
    os.environ.setdefault("KERAS_BACKEND", backend)
    return os.environ["KERAS_BACKEND"]


# Side effects on import (order matters: both must run before any jax/keras import).
ACTIVE_GPUS = limit_gpus()
ACTIVE_BACKEND = set_backend()
