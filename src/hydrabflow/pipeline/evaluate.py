"""Stage 3: evaluation on a simulated test set (with known ground truth).

Loads the trained approximator + fitted preprocessing from ``cfg.model_dir``, draws posterior
samples for a held-out simulated dataset, and computes truth-aware diagnostics (RMSE +
calibration metrics, recovery, simulation-based calibration ECDF, z-score contraction).

Three paths, selected by ``composition.level``:

* ``none`` — single-level: sample every test row independently (the template default).
* ``global`` — compositional: the test set is a grouped dataset (``simulate_multistream``);
  the group members are pooled with ``compositional_sample`` using the simulator's prior score
  (generalizes ``main_eval_new_rotationcurve_agama.py``).
* ``local`` — per-member evaluation of the local model, conditioned on the *true* globals
  (generalizes ``main_eval_local_nocomposition_*``); posterior samples and targets are mapped
  back to physical units through the preprocessing inverse, and diagnostics are written per
  stream.
"""

from __future__ import annotations

import json
import os

import numpy as np

from hydrabflow.pipeline import io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.adapter import select_adapter_keys
from hydrabflow.pipeline.checkpoint import load_approximator
from hydrabflow.pipeline.compositional import (
    apply_augmentations_once,
    composition_level,
    condition_keys,
    flatten_members,
    group_members,
    log10_keys_from_pipeline,
    prior_score_from_spec,
)
from hydrabflow.pipeline.workflow import build_workflow
from hydrabflow.preprocessing.registry import build_pipeline
from hydrabflow.utils.logging import get_logger
from hydrabflow.utils.paths import POSTERIOR_SAMPLES, PREPROCESSING_STATE, get_run_dir
from hydrabflow.utils.seed import seed_everything

log = get_logger(__name__)


def _require_model_dir(cfg) -> str:
    if not cfg.model_dir:
        raise ValueError(
            "evaluate requires `model_dir` to point at a completed training run, e.g. "
            "model_dir=outputs/<sim>/<model>/<timestamp>"
        )
    return cfg.model_dir


def _sample_kwargs(cfg) -> dict:
    from omegaconf import OmegaConf

    kw = cfg.eval.sample_kwargs
    return OmegaConf.to_container(kw, resolve=True) if OmegaConf.is_config(kw) else dict(kw)


def _load_test_data(cfg, model_dir: str):
    """Load the held-out test set and replay the *fitted* preprocessing (no re-fit, no split)."""
    test_path = os.path.join(cfg.data.data_dir, cfg.eval.test_dataset_name)
    test_data = io.load_dataset(test_path)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(model_dir, PREPROCESSING_STATE))
    test_data = pipeline.transform(test_data)
    return select_adapter_keys(test_data, cfg), pipeline


def run_evaluation(cfg):
    level = composition_level(cfg)
    if level == "global":
        return _evaluate_compositional_global(cfg)
    if level == "local":
        return _evaluate_local(cfg)

    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    model_dir = _require_model_dir(cfg)

    workflow = build_workflow(cfg)
    workflow.approximator = load_approximator(model_dir)
    test_data, _ = _load_test_data(cfg, model_dir)

    posterior = workflow.sample(
        num_samples=int(cfg.eval.num_samples),
        conditions=test_data,
        batch_size=int(cfg.eval.batch_size),
    )
    _save_posterior(posterior, run_dir)

    param_names = list(cfg.adapter.inference_variables)
    _run_diagnostics(cfg, posterior, test_data, param_names, run_dir)
    _write_report(cfg, run_dir, model_dir, "none")
    log.info("Evaluation complete. Artifacts in %s", run_dir)
    return posterior


def _evaluate_compositional_global(cfg):
    """Global-level evaluation on a grouped (multistream) test set, both ways:

    * **base** — ordinary ``workflow.sample()`` on each group member independently (no pooling
      across the exchangeable members of a group); generalizes the reference's "nocomposition"
      eval script.
    * **compositional** — ``compositional_sample()`` pooling all members of a group with the
      simulator's prior score; generalizes the reference's compositional eval script.

    Both read the same grouped test set; outputs are prefixed accordingly.
    """
    from hydrabflow.simulators.registry import get_simulator

    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    model_dir = _require_model_dir(cfg)

    workflow = build_workflow(cfg)
    workflow.approximator = load_approximator(model_dir)
    test_data, pipeline = _load_test_data(cfg, model_dir)

    context = next(iter(cfg.adapter.inference_conditions), "j")
    n, m = np.asarray(test_data[context]).shape[:2]

    # Augment flat member rows once (fixed draw); reused for both eval modes.
    flat = apply_augmentations_once(flatten_members(test_data, m), cfg, pipeline, int(cfg.seed))
    param_names = list(cfg.adapter.inference_variables)
    # log10_transform (if configured) is undone only for diagnostics/plots below — the *saved*
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
    _save_posterior(base_posterior, run_dir, name=f"base_{POSTERIOR_SAMPLES}")
    base_targets = {k: np.asarray(flat[k]) for k in param_names}
    _run_diagnostics(
        cfg,
        pipeline.inverse_transform(dict(base_posterior)),
        pipeline.inverse_transform(dict(base_targets)),
        param_names, run_dir, prefix="base_",
    )

    # --- compositional: pool members of each group with the simulator's prior score ---
    grouped = group_members(flat, n, m)
    conditions = {k: grouped[k] for k in condition_keys(cfg) if k in grouped}

    simulator = get_simulator(cfg.simulator)
    prior_score = prior_score_from_spec(simulator.prior_spec_global, log10_keys=log10_keys)

    log.info("Compositional (global) sampling: %d datasets x %d members", n, m)
    posterior = workflow.compositional_sample(
        num_samples=int(cfg.eval.num_samples),
        conditions=conditions,
        compute_prior_score=prior_score,
        batch_size=int(cfg.eval.batch_size),
        **_sample_kwargs(cfg),
    )
    _save_posterior(posterior, run_dir, name=f"compositional_{POSTERIOR_SAMPLES}")

    targets = {k: np.asarray(test_data[k]) for k in param_names}
    _run_diagnostics(
        cfg,
        pipeline.inverse_transform(dict(posterior)),
        pipeline.inverse_transform(dict(targets)),
        param_names, run_dir, prefix="compositional_",
    )
    _write_report(cfg, run_dir, model_dir, "global")
    log.info("Global evaluation (base + compositional) complete. Artifacts in %s", run_dir)
    return {"base": base_posterior, "compositional": posterior}


def _evaluate_local(cfg):
    """Local-level evaluation: per-member sampling conditioned on the true globals."""
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    model_dir = _require_model_dir(cfg)

    workflow = build_workflow(cfg)
    workflow.approximator = load_approximator(model_dir)
    test_data, pipeline = _load_test_data(cfg, model_dir)

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
    _save_posterior({k: v for k, v in posterior.items() if k != "j"}, run_dir)

    # Diagnostics per stream (each stream has its own local-parameter scales).
    from hydrabflow.simulators.registry import get_simulator

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
        _run_diagnostics(cfg, est, targ, param_names, run_dir, prefix=f"{name}_")
    _write_report(cfg, run_dir, model_dir, "local")
    log.info("Local evaluation complete. Artifacts in %s", run_dir)
    return posterior


def _write_report(cfg, run_dir: str, model_dir: str, level: str) -> None:
    """Best-effort ``report.md`` from the metrics/figures just written; never aborts a run."""
    try:
        from hydrabflow.utils.reporting import write_report

        write_report(
            run_dir,
            param_names=list(cfg.adapter.inference_variables),
            model_dir=model_dir,
            title=f"Evaluation report (composition={level})",
        )
    except Exception as exc:  # report generation must never abort an evaluation
        log.warning("Could not write report.md: %s", exc)


def _save_posterior(posterior, run_dir: str, name: str = POSTERIOR_SAMPLES) -> None:
    np.savez(
        os.path.join(run_dir, name),
        **{k: np.asarray(v) for k, v in posterior.items()},
    )


def _run_diagnostics(cfg, posterior, test_data, param_names, run_dir, prefix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import bayesflow as bf
    from bayesflow.diagnostics import metrics as bf_metrics

    requested = list(cfg.eval.diagnostics)

    if "metrics" in requested:
        try:
            results = {}
            for name, fn in (
                ("rmse", bf_metrics.root_mean_squared_error),
                ("calibration_error", bf_metrics.calibration_error),
            ):
                out = fn(estimates=posterior, targets=test_data, variable_keys=param_names)
                results[name] = {
                    "values": np.asarray(out["values"]).tolist(),
                    "mean": float(np.mean(out["values"])),
                }
            with open(os.path.join(run_dir, f"{prefix}metrics.json"), "w") as f:
                json.dump(results, f, indent=2)
            log.info("Metrics%s: %s", f" ({prefix.rstrip('_')})" if prefix else "",
                     {k: v["mean"] for k, v in results.items()})
        except Exception as exc:
            log.warning("metrics failed: %s", exc)

    plot_specs = [
        ("recovery", getattr(bf.diagnostics, "recovery", None)),
        ("calibration_ecdf", getattr(bf.diagnostics, "calibration_ecdf", None)),
        ("coverage", getattr(bf.diagnostics, "coverage", None)),
        ("z_score_contraction", getattr(bf.diagnostics, "z_score_contraction", None)),
    ]
    for name, fn in plot_specs:
        if name not in requested or fn is None:
            continue
        try:
            fig = fn(estimates=posterior, targets=test_data, variable_names=param_names)
            fig.savefig(os.path.join(run_dir, f"{prefix}{name}.png"), bbox_inches="tight")
        except Exception as exc:
            log.warning("%s failed: %s", name, exc)


cli = make_cli(run_evaluation)


if __name__ == "__main__":
    cli()
