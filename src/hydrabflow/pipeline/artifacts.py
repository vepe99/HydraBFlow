"""What a stage writes into its run directory: model, loss curve, posterior, diagnostics.

Shared by train/evaluate/tune, so no stage imports another's internals. Plot and metric helpers are
best-effort — a failed figure must never abort a run.

``load_approximator`` carries a BayesFlow workaround: a JAX-backed approximator serializes array
constants tagged ``__bayesflow_type__ArrayImpl``, which can fail to reload, so we patch the tag to
``__bayesflow_type__ndarray`` in a copy of the archive first.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from typing import Any

import matplotlib
import numpy as np

from hydrabflow.utils.paths import (
    CONVERGENCE_JSON,
    HISTORY_JSON,
    LOSS_PLOT,
    MODEL_FILENAME,
    POSTERIOR_SAMPLES,
)

matplotlib.use("Agg")  # headless: runs land in a run dir, never on a screen

log = logging.getLogger(__name__)


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
    # compile=False: we only sample here, never resume training. Restoring the saved optimizer state
    # is useless and can hard-fail (its slot variables need not match the rebuilt network's, e.g.
    # TimeSeriesTransformer's time2vec weights).
    return keras.models.load_model(fix_keras_model(path), compile=False)


def save_posterior(posterior, run_dir: str, name: str = POSTERIOR_SAMPLES) -> None:
    np.savez(
        os.path.join(run_dir, name),
        **{k: np.asarray(v) for k, v in posterior.items()},
    )


def save_history(history, run_dir: str) -> None:
    """Persist the raw Keras loss history as JSON, a loss curve, and a convergence report.

    The report (``convergence.json``: NaN / overfitting / under-training flags) is worth having
    before a chained evaluation stage runs; good training loss does not imply good inference, but a
    diverged or overfit run is worth flagging early.
    """
    hist = getattr(history, "history", None) or {}
    try:
        with open(os.path.join(run_dir, HISTORY_JSON), "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in hist.items()}, f, indent=2)
    except Exception as exc:
        log.warning("Could not save %s: %s", HISTORY_JSON, exc)

    try:
        import bayesflow as bf

        bf.diagnostics.plots.loss(history).savefig(
            os.path.join(run_dir, LOSS_PLOT), bbox_inches="tight"
        )
    except Exception as exc:
        log.warning("Could not save loss plot: %s", exc)

    try:
        from hydrabflow.utils.reporting import inspect_history

        report = inspect_history(hist)
        with open(os.path.join(run_dir, CONVERGENCE_JSON), "w") as f:
            json.dump(report, f, indent=2)
        if report["overall"]["ok"]:
            log.info("Training convergence: OK (no NaN / overfitting / under-training).")
        else:
            for issue in report["overall"]["issues"]:
                log.warning("Convergence check: %s", issue)
    except Exception as exc:
        log.warning("Could not save convergence report: %s", exc)


def restore_best_weights(workflow, run_dir: str) -> None:
    """Load the best-val-loss weights BayesFlow checkpointed during training.

    Makes the persisted model the best one seen rather than the last epoch's (which may be worse,
    or NaN after a late divergence). A missing checkpoint leaves the in-memory weights untouched.
    """
    from hydrabflow.pipeline.workflow import BEST_WEIGHTS_NAME

    ckpt = os.path.join(run_dir, BEST_WEIGHTS_NAME + ".weights.h5")
    if not os.path.exists(ckpt):
        log.info("No best-weights checkpoint at %s; keeping final-epoch weights.", ckpt)
        return
    try:
        workflow.approximator.load_weights(ckpt)
        log.info("Restored best-val-loss weights from %s", ckpt)
    except Exception as exc:
        log.warning("Could not restore best weights from %s: %s", ckpt, exc)


def run_diagnostics(cfg, posterior, targets, param_names, run_dir: str, prefix: str = "") -> None:
    """Write the truth-aware diagnostics listed in ``cfg.eval.diagnostics`` into ``run_dir``.

    ``targets`` must carry ground-truth parameters, so this applies to a simulated test set or a
    validation split — not to real data. ``prefix`` distinguishes several evaluations written into
    one run dir (e.g. ``base_`` vs ``compositional_`` at the global composition level).
    """
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
                out = fn(estimates=posterior, targets=targets, variable_keys=param_names)
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

    for name in ("recovery", "calibration_ecdf", "coverage", "z_score_contraction"):
        fn = getattr(bf.diagnostics, name, None)
        if name not in requested or fn is None:
            continue
        try:
            fig = fn(estimates=posterior, targets=targets, variable_names=param_names)
            fig.savefig(os.path.join(run_dir, f"{prefix}{name}.png"), bbox_inches="tight")
        except Exception as exc:
            log.warning("%s failed: %s", name, exc)


def save_posterior_plot(posterior, param_names, run_dir: str) -> None:
    """Truth-free diagnostic: one posterior pair plot per observation (used for real data)."""
    import bayesflow as bf

    fn = getattr(bf.diagnostics, "pairs_posterior", None) or getattr(
        bf.diagnostics, "pairs_samples", None
    )
    if fn is None:
        log.warning("No posterior pair-plot helper found in bayesflow.diagnostics.")
        return
    n_obs = int(np.asarray(next(iter(posterior.values()))).shape[0])
    for i in range(n_obs):
        try:
            single = {k: np.asarray(v)[i] for k, v in posterior.items()}
            fig = fn(estimates=single, variable_names=param_names)
            suffix = "" if n_obs == 1 else f"_obs{i}"
            fig.savefig(os.path.join(run_dir, f"posterior_pairs{suffix}.png"), bbox_inches="tight")
        except Exception as exc:
            log.warning("posterior plot for observation %d failed: %s", i, exc)


def write_report(cfg, run_dir: str, model_dir: str, title: str) -> None:
    """Best-effort ``report.md`` from the metrics/figures just written; never aborts a run."""
    try:
        from hydrabflow.utils.reporting import write_report as _write

        _write(
            run_dir,
            param_names=list(cfg.adapter.inference_variables),
            model_dir=model_dir,
            title=title,
        )
    except Exception as exc:
        log.warning("Could not write report.md: %s", exc)
