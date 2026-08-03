"""Artifacts a stage writes into a run directory: loss curve, posterior, diagnostics.

Shared by train, evaluate, and tune so no stage has to import another stage's internals. Every
helper is best-effort: a failed plot or metric must never abort a training or evaluation run.
"""

from __future__ import annotations

import json
import logging
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: runs land in a run dir, never on a screen

log = logging.getLogger(__name__)


def save_posterior(posterior, run_dir: str) -> None:
    np.savez(
        os.path.join(run_dir, "posterior.npz"),
        **{k: np.asarray(v) for k, v in posterior.items()},
    )


def save_history(history, run_dir: str) -> None:
    """Persist the raw Keras loss history as JSON, plus a loss curve."""
    hist = getattr(history, "history", None) or {}
    try:
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in hist.items()}, f, indent=2)
    except Exception as exc:
        log.warning("Could not save history.json: %s", exc)

    try:
        import bayesflow as bf

        bf.diagnostics.plots.loss(history).savefig(
            os.path.join(run_dir, "loss.png"), bbox_inches="tight"
        )
    except Exception as exc:
        log.warning("Could not save loss plot: %s", exc)


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


def run_diagnostics(cfg, posterior, targets, param_names, run_dir: str) -> None:
    """Write the truth-aware diagnostics listed in ``cfg.eval.diagnostics`` into ``run_dir``.

    ``targets`` must carry ground-truth parameters, so this applies to a simulated test set or a
    validation split — not to real data.
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
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(results, f, indent=2)
            log.info("Metrics: %s", {k: v["mean"] for k, v in results.items()})
        except Exception as exc:
            log.warning("metrics failed: %s", exc)

    for name in ("recovery", "calibration_ecdf", "coverage", "z_score_contraction"):
        fn = getattr(bf.diagnostics, name, None)
        if name not in requested or fn is None:
            continue
        try:
            fig = fn(estimates=posterior, targets=targets, variable_names=param_names)
            fig.savefig(os.path.join(run_dir, f"{name}.png"), bbox_inches="tight")
        except Exception as exc:
            log.warning("%s failed: %s", name, exc)


def save_posterior_plot(posterior, param_names, run_dir: str) -> None:
    """Truth-free diagnostic: one posterior pair plot per observation (used for real data)."""
    try:
        import bayesflow as bf

        fn = getattr(bf.diagnostics, "pairs_posterior", None) or getattr(
            bf.diagnostics, "pairs_samples", None
        )
        if fn is None:
            log.warning("No posterior pair-plot helper found in bayesflow.diagnostics.")
            return
        n_obs = int(np.asarray(next(iter(posterior.values()))).shape[0])
        for i in range(n_obs):
            single = {k: np.asarray(v)[i] for k, v in posterior.items()}
            fig = fn(estimates=single, variable_names=param_names)
            suffix = "" if n_obs == 1 else f"_obs{i}"
            fig.savefig(os.path.join(run_dir, f"posterior_pairs{suffix}.png"), bbox_inches="tight")
    except Exception as exc:
        log.warning("posterior plot failed: %s", exc)
