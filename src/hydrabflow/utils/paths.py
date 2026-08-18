"""Run-directory helpers."""

from __future__ import annotations

import logging
import os
import shutil

log = logging.getLogger(__name__)

#: Fitted preprocessing state, written by train and reloaded by evaluate (hence not a local literal).
PREPROCESSING_STATE = "preprocessing_state.npz"


def get_run_dir() -> str:
    """The current Hydra run output dir (works regardless of the ``job.chdir`` setting).

    This is ``hydra.run.dir`` = ``outputs/${simulator.name}/${run_name}/<timestamp>``, and it is the
    *only* output location the config exposes: there is deliberately no per-stage ``output_dir`` key.
    Hydra writes the resolved config into its ``.hydra/`` subfolder, so artifacts written here are
    automatically co-located with the config that produced them.

    The two exceptions are paths that must outlive a single launch and therefore *are* configured
    explicitly: datasets (``data.data_dir``, see ``save_config_snapshot``) and the Optuna study plus
    trial artifacts (``tuning.storage_dir`` / ``tuning.artifacts_dir``, which N parallel launches
    share). Everything else -- the trained approximator, preprocessing state, posteriors,
    diagnostics, ``best_trials.json`` -- lands in this dir.
    """
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
