"""Stage 3: posterior inference on held-out data, simulated or real.

Loads the approximator + fitted preprocessing from ``cfg.model_dir``, then samples. Two modes, one
entry point: the simulated test set (has ground truth -> truth-aware diagnostics), or
``data.real_data_path=<your.npz>`` (no truth, no resimulation -> posterior pair plots only; the
real-data paths live in :mod:`evaluate_real`).

``composition.level`` selects how the simulated set is scored:

* ``none`` — single-level: sample every test row independently (the template default).
* ``global`` — the test set is grouped (``simulate_multistream``); members are scored both
  independently (``base_`` prefix) and pooled with ``compositional_sample`` + the simulator's
  prior score (``compositional_`` prefix).
* ``local`` — per-member evaluation of the local model, conditioned on the *true* globals;
  posteriors and targets are mapped back to physical units through the preprocessing inverse and
  diagnostics are written per stream.

``model_dir`` is read-only: results go to this launch's own Hydra run dir (``get_run_dir()``), not
back into ``model_dir``, so one model can be evaluated any number of times without clobbering.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from hydrabflow.pipeline import artifacts, io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.adapter import composition_level, select_adapter_keys
from hydrabflow.pipeline.compositional import (
    apply_augmentations_once,
    build_prior_score,
    apply_mask_plan,
    apply_observed_groups,
    condition_keys,
    flatten_members,
    group_members,
    log10_keys_from_pipeline,
    sample_kwargs,
)
from hydrabflow.pipeline.misspecification import save_member_summaries
from hydrabflow.pipeline.workflow import build_workflow
from hydrabflow.registry import build_pipeline, get_simulator
from hydrabflow.utils.paths import POSTERIOR_SAMPLES, PREPROCESSING_STATE, get_run_dir
from hydrabflow.utils.seed import seed_everything

log = logging.getLogger(__name__)


def _load_model(cfg):
    """Rebuild the workflow and load the trained approximator from ``cfg.model_dir``."""
    if not cfg.model_dir:
        raise ValueError(
            "evaluate requires `model_dir` to point at a completed training run, e.g. "
            "model_dir=outputs/<sim>/<run>/<timestamp>"
        )
    workflow = build_workflow(cfg)
    workflow.approximator = artifacts.load_approximator(cfg.model_dir)
    # Applies to every ordinary `sample` call below, whichever path runs; the compositional mask
    # plan sets its own per-item mask and takes precedence where both are configured.
    apply_observed_groups(workflow, cfg)
    return workflow


def _load_test_data(cfg, model_dir: str):
    """Load the held-out test set and replay the *fitted* preprocessing (no re-fit, no split).

    Returns ``(test_data, pipeline)``; the offline diagnostics scripts reuse this exact path.
    """
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(model_dir, PREPROCESSING_STATE))
    path = os.path.join(cfg.data.data_dir, cfg.eval.test_dataset_name)
    return select_adapter_keys(pipeline.transform(io.load_dataset(path)), cfg), pipeline


def run_evaluation(cfg):
    if cfg.data.real_data_path:
        from hydrabflow.pipeline.evaluate_real import run_real_evaluation

        return run_real_evaluation(cfg)

    level = composition_level(cfg)
    if level == "global":
        return _evaluate_compositional_global(cfg)
    if level == "local":
        return _evaluate_local(cfg)

    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    workflow = _load_model(cfg)
    test_data, pipeline = _load_test_data(cfg, cfg.model_dir)

    posterior = workflow.sample(
        num_samples=int(cfg.eval.num_samples),
        conditions=test_data,
        batch_size=int(cfg.eval.batch_size),
    )
    param_names = list(cfg.adapter.inference_variables)
    artifacts.save_posterior(posterior, run_dir)
    artifacts.run_diagnostics(cfg, posterior, test_data, param_names, run_dir)
    artifacts.write_report(cfg, run_dir, cfg.model_dir, "Evaluation report (composition=none)")
    log.info("Evaluation complete. Artifacts in %s", run_dir)
    return posterior


def _evaluate_compositional_global(cfg):
    """Global-level evaluation on a grouped (multistream) test set, both ways:

    * **base** — ordinary ``workflow.sample()`` on each group member independently;
    * **compositional** — ``compositional_sample()`` pooling all members of a group with the
      simulator's prior score.
    """
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    workflow = _load_model(cfg)
    test_data, pipeline = _load_test_data(cfg, cfg.model_dir)

    context = next(iter(cfg.adapter.inference_conditions), "j")
    n, m = np.asarray(test_data[context]).shape[:2]

    # Augment flat member rows once (fixed draw); reused for both eval modes.
    flat = apply_augmentations_once(flatten_members(test_data, m), cfg, pipeline, int(cfg.seed))
    # Best-effort reference set for the real-data misspecification test (never aborts).
    save_member_summaries(workflow.approximator, flat, run_dir)
    param_names = list(cfg.adapter.inference_variables)
    # log10_transform (if configured) is undone only for diagnostics/plots — the *saved*
    # posterior.npz stays in the model's native space, since a real-data local-level evaluation
    # may chain off it as ancestral conditions (which must match training-time units).
    log10_keys = log10_keys_from_pipeline(pipeline)

    # --- base: ordinary per-member sampling, no pooling ---
    flat_conditions = {k: flat[k] for k in condition_keys(cfg) if k in flat}
    log.info("Base (non-compositional) sampling: %d rows", n * m)
    base_posterior = workflow.sample(
        num_samples=int(cfg.eval.num_samples),
        conditions=flat_conditions,
        batch_size=int(cfg.eval.batch_size),
    )
    artifacts.save_posterior(base_posterior, run_dir, name=f"base_{POSTERIOR_SAMPLES}")
    base_targets = {k: np.asarray(flat[k]) for k in param_names}
    artifacts.run_diagnostics(
        cfg,
        pipeline.inverse_transform(dict(base_posterior)),
        pipeline.inverse_transform(dict(base_targets)),
        param_names, run_dir, prefix="base_",
    )

    # --- compositional: pool members of each group with the simulator's prior score ---
    if not hasattr(workflow, "compositional_sample"):
        log.info(
            "Workflow has no compositional_sample (inference net is not a DiffusionModel): "
            "skipping the compositional stage, base_* metrics only."
        )
        artifacts.write_report(
            cfg, run_dir, cfg.model_dir, "Evaluation report (composition=global, base only)"
        )
        return {"base": base_posterior}
    grouped = group_members(flat, n, m)
    conditions = {k: grouped[k] for k in condition_keys(cfg) if k in grouped}
    prior_score = build_prior_score(
        cfg, get_simulator(cfg.simulator), log10_keys=log10_keys, param_order=param_names,
        seed=int(cfg.seed),
    )

    # Mask the group-level observables out of the per-member items and carry them in their own
    # item, so each likelihood enters the compositional product exactly once (no-op unless
    # eval.member_groups is set).
    conditions = apply_mask_plan(workflow, cfg, conditions, m)
    n_items = np.asarray(next(iter(conditions.values()))).shape[1]
    log.info("Compositional (global) sampling: %d datasets x %d items", n, n_items)
    posterior = workflow.compositional_sample(
        num_samples=int(cfg.eval.num_samples),
        conditions=conditions,
        compute_prior_score=prior_score,
        batch_size=int(cfg.eval.batch_size),
        **sample_kwargs(cfg),
    )
    artifacts.save_posterior(posterior, run_dir, name=f"compositional_{POSTERIOR_SAMPLES}")
    targets = {k: np.asarray(test_data[k]) for k in param_names}
    artifacts.run_diagnostics(
        cfg,
        pipeline.inverse_transform(dict(posterior)),
        pipeline.inverse_transform(dict(targets)),
        param_names, run_dir, prefix="compositional_",
    )
    artifacts.write_report(cfg, run_dir, cfg.model_dir, "Evaluation report (composition=global)")
    log.info("Global evaluation (base + compositional) complete. Artifacts in %s", run_dir)
    return {"base": base_posterior, "compositional": posterior}


def _evaluate_local(cfg):
    """Local-level evaluation: per-member sampling conditioned on the true globals."""
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    workflow = _load_model(cfg)
    test_data, pipeline = _load_test_data(cfg, cfg.model_dir)

    j = np.asarray(test_data["j"])
    if j.ndim == 3:  # grouped (multistream) test set -> flat member rows
        flat = flatten_members(test_data, j.shape[1])
    else:
        flat = {k: np.asarray(v) for k, v in test_data.items()}
    flat = apply_augmentations_once(flat, cfg, pipeline, int(cfg.seed))
    conditions = {k: flat[k] for k in condition_keys(cfg) if k in flat}

    log.info("Local-level sampling on %d member rows", len(np.asarray(flat["j"])))
    posterior = workflow.sample(
        num_samples=int(cfg.eval.num_samples),
        conditions=conditions,
        batch_size=int(cfg.eval.batch_size),
    )

    # Posterior samples (and targets) back to physical units via the preprocessing inverse
    # (per-stream prior normalization of the local parameters is invertible given `j`).
    param_names = list(cfg.adapter.inference_variables)
    posterior = dict(posterior)
    posterior["j"] = flat["j"]
    posterior = pipeline.inverse_transform(posterior)
    targets = pipeline.inverse_transform(
        {**{k: flat[k] for k in param_names if k in flat}, "j": flat["j"]}
    )
    artifacts.save_posterior({k: v for k, v in posterior.items() if k != "j"}, run_dir)

    # Diagnostics per stream (each stream has its own local-parameter scales).
    target_streams = getattr(get_simulator(cfg.simulator), "target_streams", None)
    stream_ids = np.asarray(flat["j"]).reshape(-1).astype(int)
    names = (
        {int(v): str(k) for k, v in target_streams.items()}
        if target_streams
        else {int(s): f"stream_{int(s)}" for s in np.unique(stream_ids)}
    )
    for stream, name in sorted(names.items()):
        rows = stream_ids == stream
        if not rows.any():
            continue
        est = {k: np.asarray(posterior[k])[rows] for k in param_names}
        targ = {k: np.asarray(targets[k])[rows] for k in param_names}
        artifacts.run_diagnostics(cfg, est, targ, param_names, run_dir, prefix=f"{name}_")
    artifacts.write_report(cfg, run_dir, cfg.model_dir, "Evaluation report (composition=local)")
    log.info("Local evaluation complete. Artifacts in %s", run_dir)
    return posterior


cli = make_cli(run_evaluation)


if __name__ == "__main__":
    cli()
