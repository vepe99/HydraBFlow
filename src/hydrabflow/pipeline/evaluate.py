"""Stage 3: evaluation on a simulated test set (with known ground truth).

Loads the trained approximator + fitted preprocessing from ``cfg.model_dir``, draws posterior
samples for a held-out simulated dataset, and computes truth-aware diagnostics (RMSE +
calibration metrics, recovery, simulation-based calibration ECDF, z-score contraction).
Generalizes ``main_eval_new_rotationcurve_agama.py`` minus the compositional reshaping.
"""

from __future__ import annotations

import json
import os

import numpy as np

from hydrabflow.pipeline import io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.checkpoint import load_approximator
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


def run_evaluation(cfg):
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    model_dir = _require_model_dir(cfg)

    # 1. Rebuild the workflow and load the trained approximator into it.
    workflow = build_workflow(cfg)
    workflow.approximator = load_approximator(model_dir)

    # 2. Load the held-out test set and replay the *fitted* preprocessing (no re-fit, no split).
    test_path = os.path.join(cfg.data.data_dir, cfg.eval.test_dataset_name)
    test_data = io.load_dataset(test_path)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(model_dir, PREPROCESSING_STATE))
    test_data = pipeline.transform(test_data)

    # 3. Sample the posterior for every test observation.
    posterior = workflow.sample(
        num_samples=int(cfg.eval.num_samples),
        conditions=test_data,
        batch_size=int(cfg.eval.batch_size),
    )
    np.savez(os.path.join(run_dir, POSTERIOR_SAMPLES), **{k: np.asarray(v) for k, v in posterior.items()})

    # 4. Diagnostics (each guarded so one failure doesn't abort the rest).
    param_names = list(cfg.adapter.inference_variables)
    _run_diagnostics(cfg, posterior, test_data, param_names, run_dir)
    _write_report(cfg, run_dir, model_dir, "none")
    log.info("Evaluation complete. Artifacts in %s", run_dir)
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


def _run_diagnostics(cfg, posterior, test_data, param_names, run_dir) -> None:
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
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(results, f, indent=2)
            log.info("Metrics: %s", {k: v["mean"] for k, v in results.items()})
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
            fig.savefig(os.path.join(run_dir, f"{name}.png"), bbox_inches="tight")
        except Exception as exc:
            log.warning("%s failed: %s", name, exc)


cli = make_cli(run_evaluation)


if __name__ == "__main__":
    cli()
