"""Stage 4: hyperparameter tuning with Optuna.

A multi-objective study (RMSE + calibration error by default). The dataset is loaded and
preprocessed once; each trial overlays its sampled hyperparameters on a copy of the config, trains a
fresh workflow for ``tuning.n_epochs``, and scores the validation split.

Concurrency: the study is a ``JournalStorage`` over one append-safe ``.log``, so launching the same
command N times (same ``study_name`` + ``storage_dir``) runs trials of one shared study.

Artifacts: with ``tuning.save_artifacts``, each trial writes its model/posterior/diagnostics to
``${tuning.artifacts_dir}/trials/trial_<number>/`` (study-global number, so no collisions); the
shared preprocessing state is saved once alongside.
"""

from __future__ import annotations

import copy
import json
import logging
import os

import numpy as np

from hydrabflow.registry import build_augmentations
from hydrabflow.pipeline import artifacts, io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.adapter import select_adapter_keys
from hydrabflow.pipeline.workflow import build_workflow
from hydrabflow.registry import build_pipeline
from hydrabflow.utils.oom import is_oom_error, run_with_oom_backoff
from hydrabflow.utils.paths import PREPROCESSING_STATE, get_run_dir
from hydrabflow.utils.seed import seed_everything

log = logging.getLogger(__name__)

# Finite worst-case RMSE for a trial that diverged to NaN: strictly dominated on the Pareto front,
# but unlike a NaN objective (which Optuna rejects) it still gives the sampler a signal.
_NAN_PENALTY_RMSE = 1.0e3

# Training a tuning trial at a tiny batch is too slow to be worth it, so a trial that still OOMs at
# this batch size is pruned rather than crawled through.
_MIN_TRIAL_BATCH = 512


def _suggest(trial, name, spec):
    t = spec["type"]
    if t == "int":
        return trial.suggest_int(
            name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
        )
    if t == "float":
        return trial.suggest_float(
            name, float(spec["low"]), float(spec["high"]),
            step=spec.get("step"), log=bool(spec.get("log", False)),
        )
    if t == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unknown search-space type '{t}' for '{name}'")


def _save_shared_preprocessing(pipeline, artifacts_dir: str) -> str:
    """Save the fit-once preprocessing state, shared by every trial.

    Written atomically and only if absent, so concurrent processes (which fit identical state from
    the same seed) don't clobber each other.
    """
    path = os.path.join(artifacts_dir, PREPROCESSING_STATE)
    if os.path.exists(path):
        return path
    tmp = f"{path}.{os.getpid()}.tmp.npz"  # np.savez appends .npz unless the name has it
    pipeline.save(tmp)
    os.replace(tmp, path)
    return path


def _objective(trial, base_cfg, train_data, val_data, param_names, augmentations):
    import keras
    import optuna
    from bayesflow.diagnostics import metrics as bf_metrics
    from omegaconf import OmegaConf

    cfg = copy.deepcopy(base_cfg)
    for path, spec in base_cfg.tuning.search_space.items():
        OmegaConf.update(cfg, path, _suggest(trial, path, spec), force_add=True)

    # Per-trial dir up front so BayesFlow checkpoints best-val-loss weights into it: exactly as the
    # train stage does, the trial is scored on its best weights, not a possibly-NaN final epoch.
    trial_dir = os.path.join(cfg.tuning.artifacts_dir, "trials", f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)
    workflow = build_workflow(cfg, run_dir=trial_dir)

    def _fit(batch_size: int):
        return workflow.fit_offline(
            train_data,
            validation_data=val_data,
            epochs=int(cfg.tuning.n_epochs),
            batch_size=int(batch_size),
            augmentations=augmentations or None,
            verbose=0,
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                artifacts.epoch_log_callback(log, prefix=f"trial {trial.number}: "),
            ],
        )

    try:
        history = run_with_oom_backoff(
            _fit, int(cfg.training.batch_size), min_batch=_MIN_TRIAL_BATCH, logger=log
        )
    except Exception as exc:
        if is_oom_error(exc):
            log.warning("Trial %d: OOM at batch_size<=%d; pruning.", trial.number,
                        _MIN_TRIAL_BATCH)
            raise optuna.TrialPruned() from exc
        raise
    artifacts.restore_best_weights(workflow, trial_dir)

    posterior = run_with_oom_backoff(
        lambda batch_size: workflow.sample(
            num_samples=int(cfg.eval.num_samples),
            conditions=val_data,
            batch_size=int(batch_size),
        ),
        int(cfg.eval.batch_size),
        logger=log,
    )
    rmse_mean = float(np.mean(bf_metrics.root_mean_squared_error(
        estimates=posterior, targets=val_data, variable_keys=param_names)["values"]))
    cal_mean = float(np.mean(bf_metrics.calibration_error(
        estimates=posterior, targets=val_data, variable_keys=param_names)["values"]))

    # A trial that diverged from the start has no finite best weights, so its RMSE is NaN. Return a
    # finite worst case instead, so the trial completes and the sampler learns to avoid that region.
    if not np.isfinite(rmse_mean):
        log.warning("Trial %d: non-finite RMSE (training diverged); penalizing.", trial.number)
        rmse_mean = _NAN_PENALTY_RMSE
    if not np.isfinite(cal_mean):
        cal_mean = 1.0

    # The val split carries ground truth, so the evaluate stage's diagnostics apply here too.
    if bool(cfg.tuning.save_artifacts):
        artifacts.save_approximator(workflow, trial_dir)
        artifacts.save_history(history, trial_dir)
        artifacts.save_posterior(posterior, trial_dir)
        artifacts.run_diagnostics(cfg, posterior, val_data, param_names, trial_dir)
        trial.set_user_attr("artifact_dir", trial_dir)
        trial.set_user_attr("rmse", rmse_mean)
        trial.set_user_attr("calibration_error", cal_mean)

    return rmse_mean, cal_mean


def run_tuning(cfg):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    rng = seed_everything(cfg.seed)
    run_dir = get_run_dir()

    # Load + preprocess once; reuse across trials.
    data = io.load_config_dataset(cfg)
    pipeline = build_pipeline(cfg.preprocessing)
    train_data, val_data = pipeline.fit_transform(data, rng)
    train_data = select_adapter_keys(train_data, cfg)
    val_data = select_adapter_keys(val_data, cfg) if val_data is not None else None
    param_names = list(cfg.adapter.inference_variables)

    # Augmentations are part of the model being tuned: applied inside every trial's fit, and
    # replayed once (fixed draw) on the validation split used for scoring.
    augmentations = build_augmentations(
        cfg.augmentation, np.random.default_rng(cfg.seed), context={"pipeline": pipeline}
    )
    if augmentations and val_data is not None:
        for aug in augmentations:
            val_data = aug(val_data)
        val_data = {k: np.asarray(v) for k, v in val_data.items()}

    if bool(cfg.tuning.save_artifacts):
        os.makedirs(cfg.tuning.artifacts_dir, exist_ok=True)
        log.info("Shared preprocessing state -> %s",
                 _save_shared_preprocessing(pipeline, cfg.tuning.artifacts_dir))

    os.makedirs(cfg.tuning.storage_dir, exist_ok=True)
    log_path = os.path.join(cfg.tuning.storage_dir, cfg.tuning.study_name + ".log")
    study = optuna.create_study(
        study_name=cfg.tuning.study_name,
        storage=JournalStorage(JournalFileBackend(log_path)),
        directions=list(cfg.tuning.directions),
        load_if_exists=True,
    )
    log.info("Study '%s' storage=%s artifacts=%s", cfg.tuning.study_name, log_path,
             cfg.tuning.artifacts_dir if bool(cfg.tuning.save_artifacts) else "(disabled)")

    # catch=(Exception,): a trial that raises is marked FAIL and the worker moves on, instead of the
    # exception propagating out of optimize() and killing the whole process.
    study.optimize(
        lambda trial: _objective(trial, cfg, train_data, val_data, param_names, augmentations),
        n_trials=int(cfg.tuning.n_trials),
        catch=(Exception,),
    )

    _report(study, cfg, run_dir)
    return study


def _report(study, cfg, run_dir) -> None:
    best = [
        {
            "number": t.number,
            "values": t.values,
            "params": t.params,
            "artifact_dir": t.user_attrs.get("artifact_dir"),
        }
        for t in study.best_trials
    ]
    # This process's run dir (traceability) plus the shared artifacts dir (so concurrent processes
    # all leave a copy next to the trial folders).
    targets = [run_dir] + ([cfg.tuning.artifacts_dir] if bool(cfg.tuning.save_artifacts) else [])
    for d in targets:
        with open(os.path.join(d, "best_trials.json"), "w") as f:
            json.dump(best, f, indent=2)
    log.info("Tuning complete: %d Pareto/best trial(s). See %s", len(best), run_dir)


cli = make_cli(run_tuning)


if __name__ == "__main__":
    cli()
