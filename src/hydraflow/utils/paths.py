"""Run-directory helpers and shared artifact filenames."""

from __future__ import annotations

import os
import shutil

from hydraflow.utils.logging import get_logger

log = get_logger(__name__)

# Filenames written into every run dir (alongside Hydra's automatic `.hydra/`).
PREPROCESSING_STATE = "preprocessing_state.npz"
POSTERIOR_SAMPLES = "posterior.npz"
LOSS_PLOT = "loss.png"


def get_run_dir() -> str:
    """Return the current Hydra run output dir (works regardless of the ``job.chdir`` setting)."""
    from hydra.core.hydra_config import HydraConfig

    return HydraConfig.get().runtime.output_dir


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_config_snapshot(dest_dir: str, stem: str | None = None) -> str | None:
    """Copy Hydra's auto-generated ``.hydra/`` config folder next to a generated artifact.

    The Hydra run dir (``outputs/...``) always gets a ``.hydra/`` snapshot, but datasets are
    written to ``data.data_dir`` instead, so they carry no record of the config that produced
    them. This copies that snapshot into ``dest_dir`` so a dataset can always be traced back to
    its exact configuration (full traceability principle).

    ``stem`` keys the snapshot to a specific file (e.g. ``training_data_10000``) so multiple
    datasets written to the same ``data_dir`` (training vs. test) don't overwrite each other's
    config. Without it, a plain ``.hydra/`` folder is written. Returns the snapshot path, or
    ``None`` if Hydra's source ``.hydra/`` could not be located.
    """
    src = os.path.join(get_run_dir(), ".hydra")
    if not os.path.isdir(src):
        log.warning("No Hydra .hydra/ folder found at %s; skipping config snapshot.", src)
        return None
    dest = os.path.join(dest_dir, f"{stem}.hydra" if stem else ".hydra")
    os.makedirs(dest_dir, exist_ok=True)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest)
    return dest
