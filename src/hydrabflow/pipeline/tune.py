"""Stage 4: hyperparameter tuning with Optuna.

Runs a (by default multi-objective) study minimizing RMSE and calibration error. The dataset is
loaded and preprocessed once; each trial applies its sampled hyperparameters onto a copy of the
config, builds a fresh workflow, trains for a short budget (``tuning.n_epochs``), and scores the
validation split. Generalizes ``main_hyperparameter_tuning_nocompositional_rotationcurve_agama.py``
but composes the config via ``compose`` rather than nesting ``@hydra.main``.

The search space (``conf/tuning/search_space``) maps dotted config paths to sampling specs:
``{type: int|float|categorical, low, high, step, log, choices}``.

**Concurrency.** The Optuna study lives in a :class:`JournalStorage` backed by a single ``.log``
file (``${tuning.storage_dir}/${tuning.study_name}.log``). Unlike the SQLite backend, the journal
file backend is safe for many processes to append to at once, so you can launch the same tuning
command N times (same ``study_name`` + ``storage_dir``) and they cooperatively run trials of one
shared study.

**Artifacts.** With ``tuning.save_artifacts`` (default ``true``) every trial persists its trained
model, posterior samples, and diagnostic plots under
``${tuning.artifacts_dir}/trials/trial_<number>/``, keyed by the *study-global* Optuna trial
number so concurrent processes never collide. The preprocessing is fit once and shared by every
trial/model, so it is saved a single time at ``${tuning.artifacts_dir}/preprocessing_state.npz``.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np

from hydrabflow.augmentation.registry import build_augmentations
from hydrabflow.pipeline import io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.adapter import select_adapter_keys
from hydrabflow.pipeline.checkpoint import save_approximator
from hydrabflow.pipeline.evaluate import _run_diagnostics
from hydrabflow.pipeline.train import _save_loss_plot
from hydrabflow.pipeline.workflow import build_workflow
from hydrabflow.preprocessing.registry import build_pipeline
from hydrabflow.utils.logging import get_logger
from hydrabflow.utils.paths import POSTERIOR_SAMPLES, PREPROCESSING_STATE, ensure_dir, get_run_dir
from hydrabflow.utils.seed import seed_everything

log = get_logger(__name__)


def _suggest(trial, name, spec):
    t = spec["type"]
    if t == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if t == "float":
        return trial.suggest_float(
            name, float(spec["low"]), float(spec["high"]),
            step=spec.get("step"), log=bool(spec.get("log", False)),
        )
    if t == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unknown search-space type '{t}' for '{name}'")


def _trial_dir(cfg, trial) -> str:
    """Per-trial artifact directory, keyed by the study-global Optuna trial number."""
    return ensure_dir(os.path.join(cfg.tuning.artifacts_dir, "trials", f"trial_{trial.number:04d}"))


def _save_shared_preprocessing(pipeline, artifacts_dir: str) -> str:
    """Save the fit-once preprocessing state, shared by every trial/model.

    Written atomically and only if absent, so concurrent processes (which fit identical state from
    the same seed) don't clobber each other.
    """
    path = os.path.join(artifacts_dir, PREPROCESSING_STATE)
    if os.path.exists(path):
        return path
    # np.savez appends ".npz" when the name lacks it, so keep the temp name ending in ".npz".
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    pipeline.save(tmp)
    os.replace(tmp, path)  # atomic on the same filesystem
    return path


def _objective(trial, base_cfg, train_data, val_data, param_names, augmentations):
    from omegaconf import OmegaConf

    cfg = copy.deepcopy(base_cfg)
    for path, spec in base_cfg.tuning.search_space.items():
        OmegaConf.update(cfg, path, _suggest(trial, path, spec), force_add=True)

    workflow = build_workflow(cfg)
    history = workflow.fit_offline(
        train_data,
        validation_data=val_data,
        epochs=int(cfg.tuning.n_epochs),
        batch_size=int(cfg.training.batch_size),
        augmentations=augmentations if augmentations else None,
        verbose=0,
    )

    from bayesflow.diagnostics import metrics as bf_metrics

    posterior = workflow.sample(
        num_samples=int(cfg.inference.num_samples),
        conditions=val_data,
        batch_size=int(cfg.inference.batch_size),
    )
    rmse = bf_metrics.root_mean_squared_error(
        estimates=posterior, targets=val_data, variable_keys=param_names
    )
    cal = bf_metrics.calibration_error(
        estimates=posterior, targets=val_data, variable_keys=param_names
    )
    rmse_mean = float(np.mean(rmse["values"]))
    cal_mean = float(np.mean(cal["values"]))

    # Persist the full trial (model + posterior + diagnostics). The val split carries ground truth,
    # so the same truth-aware diagnostics the evaluate stage produces apply here.
    if bool(cfg.tuning.save_artifacts):
        trial_dir = _trial_dir(cfg, trial)
        save_approximator(workflow, trial_dir)
        _save_loss_plot(history, trial_dir)
        np.savez(
            os.path.join(trial_dir, POSTERIOR_SAMPLES),
            **{k: np.asarray(v) for k, v in posterior.items()},
        )
        _run_diagnostics(cfg, posterior, val_data, param_names, trial_dir)
        trial.set_user_attr("artifact_dir", trial_dir)
        trial.set_user_attr("rmse", rmse_mean)
        trial.set_user_attr("calibration_error", cal_mean)

    if len(base_cfg.tuning.directions) == 1:
        return rmse_mean
    return rmse_mean, cal_mean


def run_tuning(cfg):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    rng = seed_everything(cfg.seed)
    run_dir = get_run_dir()

    # Load + preprocess once; reuse across trials.
    data = io.load_dataset(os.path.join(cfg.data.data_dir, cfg.data.dataset_name))
    pipeline = build_pipeline(cfg.preprocessing)
    train_data, val_data = pipeline.fit_transform(data, rng)
    train_data = select_adapter_keys(train_data, cfg)
    val_data = select_adapter_keys(val_data, cfg) if val_data is not None else None
    param_names = list(cfg.adapter.inference_variables)

    # Per-batch augmentations are part of the model being tuned: applied inside every trial's
    # fit, and replayed once (fixed draw) on the validation split used for scoring.
    augmentations = build_augmentations(
        cfg.augmentation, np.random.default_rng(cfg.seed), context={"pipeline": pipeline}
    )
    if augmentations and val_data is not None:
        for aug in augmentations:
            val_data = aug(val_data)
        val_data = {k: np.asarray(v) for k, v in val_data.items()}

    if bool(cfg.tuning.save_artifacts):
        ensure_dir(cfg.tuning.artifacts_dir)
        state_path = _save_shared_preprocessing(pipeline, cfg.tuning.artifacts_dir)
        log.info("Shared preprocessing state -> %s", state_path)

    # Concurrency-safe study storage: a single .log (JournalFileBackend) many processes can append
    # to, so re-running the same command (same study_name + storage_dir) extends one shared study.
    ensure_dir(cfg.tuning.storage_dir)
    log_path = os.path.join(cfg.tuning.storage_dir, cfg.tuning.study_name + ".log")
    storage = JournalStorage(JournalFileBackend(log_path))
    study = optuna.create_study(
        study_name=cfg.tuning.study_name,
        storage=storage,
        directions=list(cfg.tuning.directions),
        load_if_exists=True,
    )
    log.info("Study '%s' storage=%s artifacts=%s", cfg.tuning.study_name, log_path,
             cfg.tuning.artifacts_dir if bool(cfg.tuning.save_artifacts) else "(disabled)")
    study.optimize(
        lambda trial: _objective(trial, cfg, train_data, val_data, param_names, augmentations),
        n_trials=int(cfg.tuning.n_trials),
    )

    _report(study, cfg, run_dir)
    return study


def _report(study, cfg, run_dir) -> None:
    if len(cfg.tuning.directions) == 1:
        best_trials = [study.best_trial]
    else:
        best_trials = study.best_trials
    best = [
        {
            "number": t.number,
            "values": t.value if len(cfg.tuning.directions) == 1 else t.values,
            "params": t.params,
            "artifact_dir": t.user_attrs.get("artifact_dir"),
        }
        for t in best_trials
    ]
    # Write to this process's run dir (traceability) and the shared artifacts dir (so concurrent
    # processes all leave a copy next to the trial folders).
    targets = [run_dir]
    if bool(cfg.tuning.save_artifacts):
        targets.append(cfg.tuning.artifacts_dir)
    for d in targets:
        with open(os.path.join(d, "best_trials.json"), "w") as f:
            json.dump(best, f, indent=2)
    log.info("Tuning complete: %d Pareto/best trial(s). See %s", len(best), run_dir)


cli = make_cli(run_tuning)


if __name__ == "__main__":
    cli()
