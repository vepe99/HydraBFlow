"""Pin compute settings *before* keras/bayesflow/JAX are imported anywhere.

Two things have to happen before the first ``import jax`` / ``import keras`` in the process,
because both libraries read their configuration at import time:

1. **GPU selection.** :func:`limit_gpus` uses `autocvd <https://pypi.org/project/autocvd>`_ to pick
   available (free) GPU(s) and pin ``CUDA_VISIBLE_DEVICES`` accordingly, so JAX only ever sees
   (and pre-allocates memory on) the GPUs we intend to use. This must run before JAX initializes
   CUDA, otherwise it has no effect.
2. **Keras backend.** Keras 3 selects its compute backend from ``KERAS_BACKEND`` at import time.
   This template defaults to JAX (matching the reference project and giving fast, vectorizable
   simulators).

Importing :mod:`hydraflow` imports this module first (see ``hydraflow/__init__.py``), guaranteeing
both side effects run before the first ``import keras``/``import jax``.

Overrides (all via the environment, since this runs before any Hydra config is loaded):

- ``CUDA_VISIBLE_DEVICES`` — if you set it yourself, it is respected and autocvd is skipped.
- ``HYDRAFLOW_NUM_GPUS`` — how many GPUs autocvd should expose (default ``1``). ``0`` forces
  CPU-only by hiding all GPUs.
- ``KERAS_BACKEND`` — set to e.g. ``torch`` to override the JAX default.
"""

from __future__ import annotations

import logging
import os

DEFAULT_BACKEND = "jax"
NUM_GPUS_ENV = "HYDRAFLOW_NUM_GPUS"
DEFAULT_NUM_GPUS = 1

_log = logging.getLogger(__name__)


def limit_gpus(num_gpus: int | None = None) -> str | None:
    """Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initializes.

    Resolution order:

    1. If ``CUDA_VISIBLE_DEVICES`` is already set, respect the explicit choice and do nothing.
    2. Otherwise resolve the GPU count from ``num_gpus`` (falling back to the ``HYDRAFLOW_NUM_GPUS``
       env var, then :data:`DEFAULT_NUM_GPUS`). ``0`` (or negative) forces CPU-only by hiding all
       GPUs.
    3. Call ``autocvd`` to choose that many available GPUs and set ``CUDA_VISIBLE_DEVICES``.

    Degrades gracefully: if ``autocvd`` is not installed, or there are no NVIDIA GPUs / no
    ``nvidia-smi`` (e.g. on macOS or a CPU-only box), it logs and leaves the environment untouched.

    Returns the resulting ``CUDA_VISIBLE_DEVICES`` value, or ``None`` if it was left unset.
    """
    # Respect an explicit user choice; never override a pinned device list.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]

    if num_gpus is None:
        try:
            num_gpus = int(os.environ.get(NUM_GPUS_ENV, DEFAULT_NUM_GPUS))
        except ValueError:
            _log.warning("Invalid %s=%r; falling back to %d.", NUM_GPUS_ENV,
                         os.environ.get(NUM_GPUS_ENV), DEFAULT_NUM_GPUS)
            num_gpus = DEFAULT_NUM_GPUS

    # 0 (or negative) => force CPU-only: hide all GPUs from JAX.
    if num_gpus <= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return ""

    try:
        from autocvd import autocvd
    except ImportError:
        _log.info("autocvd not installed; not limiting GPUs (set CUDA_VISIBLE_DEVICES manually).")
        return None

    try:
        # autocvd sets CUDA_VISIBLE_DEVICES to the `num_gpus` least-utilized GPUs.
        autocvd(num_gpus=num_gpus)
    except Exception as exc:  # noqa: BLE001 — no GPUs / nvidia-smi missing (macOS, CPU box, ...)
        _log.warning("autocvd could not select GPUs (%s); leaving CUDA_VISIBLE_DEVICES unset.", exc)
        return None
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def set_backend(backend: str = DEFAULT_BACKEND) -> str:
    """Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active backend."""
    os.environ.setdefault("KERAS_BACKEND", backend)
    return os.environ["KERAS_BACKEND"]


# Side effects on import (order matters: both must run before any jax/keras import).
ACTIVE_GPUS = limit_gpus()
ACTIVE_BACKEND = set_backend()
