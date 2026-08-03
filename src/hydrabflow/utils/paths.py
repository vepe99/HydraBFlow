"""Run-directory helpers."""

from __future__ import annotations

import logging
import os
import shutil

log = logging.getLogger(__name__)

#: Fitted preprocessing state, written by train and reloaded by evaluate (hence not a local literal).
PREPROCESSING_STATE = "preprocessing_state.npz"


def get_run_dir() -> str:
    """The current Hydra run output dir (works regardless of the ``job.chdir`` setting)."""
    from hydra.core.hydra_config import HydraConfig

    return HydraConfig.get().runtime.output_dir


def save_config_snapshot(dest_dir: str, stem: str) -> str | None:
    """Copy Hydra's ``.hydra/`` config folder next to a generated dataset, keyed by its filename.

    Datasets are written to ``data.data_dir``, not the run dir, so without this they carry no record
    of the config that produced them. The ``stem`` key (e.g. ``training_data_10000``) keeps training
    and test sets in one ``data_dir`` from overwriting each other's snapshot.
    """
    src = os.path.join(get_run_dir(), ".hydra")
    if not os.path.isdir(src):
        log.warning("No Hydra .hydra/ folder found at %s; skipping config snapshot.", src)
        return None
    dest = os.path.join(dest_dir, f"{stem}.hydra")
    os.makedirs(dest_dir, exist_ok=True)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest)
    return dest
