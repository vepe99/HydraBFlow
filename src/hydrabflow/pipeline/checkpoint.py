"""Model save/load, including the BayesFlow ``.keras`` deserialization workaround.

When BayesFlow serializes a JAX-backed approximator, array constants are tagged
``__bayesflow_type__ArrayImpl`` inside the archive's ``config.json``, and reloading them can fail.
We patch the tag to ``__bayesflow_type__ndarray`` in a copy of the archive before loading.
"""

from __future__ import annotations

import logging
import os
import zipfile
from typing import Any


log = logging.getLogger(__name__)

MODEL_FILENAME = "approximator.keras"


def save_approximator(workflow: Any, run_dir: str) -> str:
    path = os.path.join(run_dir, MODEL_FILENAME)
    workflow.approximator.save(path)
    log.info("Saved approximator -> %s", path)
    return path


def fix_keras_model(model_path: str) -> str:
    """Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag)."""
    fixed = model_path.replace(".keras", "_fixed.keras")
    if os.path.exists(fixed):
        return fixed
    with zipfile.ZipFile(model_path, "r") as zin, zipfile.ZipFile(fixed, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                data = data.decode("utf-8").replace(
                    "__bayesflow_type__ArrayImpl", "__bayesflow_type__ndarray"
                ).encode("utf-8")
            zout.writestr(item, data)
    log.info("Wrote ArrayImpl-fixed model -> %s", fixed)
    return fixed


def load_approximator(run_dir: str) -> Any:
    """Load a saved approximator, applying the ArrayImpl fix first."""
    import keras

    path = os.path.join(run_dir, MODEL_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}. Train first (e.g. `hydrabflow-train`).")
    return keras.models.load_model(fix_keras_model(path))
