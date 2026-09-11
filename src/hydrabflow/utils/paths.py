"""Run-directory helpers and the artifact filenames shared between stages."""

from __future__ import annotations

import logging
import os
import shutil

log = logging.getLogger(__name__)

# Filenames written into a run dir (alongside Hydra's automatic `.hydra/`). Shared constants so a
# writer (train) and its reader (evaluate) can never drift apart.
MODEL_FILENAME = "approximator.keras"
PREPROCESSING_STATE = "preprocessing_state.npz"
POSTERIOR_SAMPLES = "posterior.npz"
LOSS_PLOT = "loss.png"
HISTORY_JSON = "history.json"          # raw Keras history.history dict
CONVERGENCE_JSON = "convergence.json"  # inspect_history() report for the training run
REPORT_MD = "report.md"                # human-readable evaluation report
SUMMARIES = "summaries.npz"            # summary-network outputs of the (augmented) member rows
MISSPECIFICATION_JSON = "misspecification.json"  # summary-space MMD test results
MMD_PLOT = "mmd_hypothesis_test.png"   # observed MMD vs bootstrap null


def get_run_dir() -> str:
    """The current Hydra run output dir (works regardless of the ``job.chdir`` setting).

    This is ``hydra.run.dir`` = ``outputs/${simulator.name}/${run_name}/<timestamp>``, and it is the
    *only* output location the config exposes: there is deliberately no per-stage ``output_dir`` key.
    Hydra writes the resolved config into its ``.hydra/`` subfolder, so artifacts written here are
    automatically co-located with the config that produced them.

    The two exceptions are paths that must outlive a single launch and therefore *are* configured
    explicitly: datasets (``data.data_dir``, see ``save_config_snapshot``) and the Optuna study plus
    trial artifacts (``tuning.storage_dir`` / ``tuning.artifacts_dir``, which N parallel launches
    share).
    """
    from hydra.core.hydra_config import HydraConfig

    return HydraConfig.get().runtime.output_dir


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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
