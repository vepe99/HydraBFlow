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

from hydrabflow.pipeline.workflow import BEST_WEIGHTS_NAME

matplotlib.use("Agg")  # headless: runs land in a run dir, never on a screen

log = logging.getLogger(__name__)

MODEL_FILENAME = "approximator.keras"


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


def load_approximator(run_dir: str, workflow=None, probe_batch=None) -> Any:
    """Load a saved approximator, applying the ArrayImpl fix first.

    ``workflow`` + ``probe_batch`` enable the fallback that a multi-branch summary network needs:
    rebuild the architecture from the config and load only the weights. A full-model load cannot
    restore one, and the cause is upstream, not in this repo -- keras walks the layer tree by
    attribute name, and for a fusion network holding its branches in a ``dict`` the save-time walk
    reaches them through the model's ``layers`` property while the load-time walk reaches them
    through the ``summary_network`` attribute. The weights are written under
    ``layers/<fusion>/backbones/...`` and looked for under ``summary_network/backbones/...``, so
    every branch reports "expected 2 variables, but received 0". Reproducible with stock
    ``bf.networks.FusionNetwork`` (``simulator=multimodal``); single-backbone models such as
    ``two_moons`` are unaffected, which is why the ``.keras`` path stays first -- it needs no probe
    batch and no config agreement.

    Rebuilding is immune because the weight file is keyed by the same construction order that wrote
    it, and the run dir carries its resolved config, so the architecture is reproducible from it.
    """
    import keras

    path = os.path.join(run_dir, MODEL_FILENAME)
    weights = os.path.join(run_dir, BEST_WEIGHTS_NAME + ".weights.h5")
    if not os.path.exists(path) and not os.path.exists(weights):
        raise FileNotFoundError(f"No saved model at {path}. Train first (e.g. `hydrabflow-train`).")

    if os.path.exists(path):
        try:
            # compile=False: we only sample here, never resume training. Restoring the saved
            # optimizer state is useless and can hard-fail (its slot variables need not match the
            # rebuilt network's, e.g. TimeSeriesTransformer's time2vec weights).
            return keras.models.load_model(fix_keras_model(path), compile=False)
        except Exception as exc:
            if workflow is None or probe_batch is None:
                raise
            log.warning("Full-model load failed (%s); rebuilding from config + weights.", exc)

    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"{path} could not be loaded and there is no {weights} to rebuild from."
        )
    # The networks build lazily, so the weights have nowhere to land until one batch has been
    # through the approximator.
    approximator = workflow.approximator
    approximator.build_from_data(approximator.adapter(probe_batch))
    approximator.load_weights(weights)
    log.info("Rebuilt approximator from config and loaded weights from %s", weights)
    return approximator


def save_posterior(posterior, run_dir: str) -> None:
    np.savez(
        os.path.join(run_dir, "posterior.npz"),
        **{k: np.asarray(v) for k, v in posterior.items()},
    )


def epoch_log_callback(logger=log, prefix: str = ""):
    """A Keras callback mirroring per-epoch metrics through ``logging``.

    Keras' progress bar writes to stdout, which Hydra's log file does not capture, so without this
    the run's ``.log`` has no train/val loss at all.
    """
    import keras

    return keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: logger.info(
            "%sEpoch %d: %s",
            prefix,
            epoch + 1,
            " ".join(f"{k}={v:.4g}" for k, v in sorted((logs or {}).items())),
        )
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

    _truth_figures(requested, posterior, targets, param_names, run_dir)

    if "per_discrete_branch" in requested:
        keys = list(cfg.adapter.inference_conditions)
        _per_branch_figures(requested, posterior, targets, param_names, run_dir, keys)


#: Below this many rows a per-branch figure is noise, not a diagnostic.
MIN_ROWS_PER_BRANCH = 50


def _truth_figures(requested, posterior, targets, param_names, run_dir, suffix: str = "") -> None:
    """The three truth-aware bf.diagnostics figures, suffixed."""
    import bayesflow as bf

    for name in ("recovery", "calibration_ecdf", "z_score_contraction"):
        fn = getattr(bf.diagnostics, name, None)
        if name not in requested or fn is None:
            continue
        try:
            fig = fn(estimates=posterior, targets=targets, variable_names=param_names)
            fig.savefig(os.path.join(run_dir, f"{name}{suffix}.png"), bbox_inches="tight")
        except Exception as exc:
            log.warning("%s%s failed: %s", name, suffix, exc)


def _mask_rows(data: dict, mask) -> dict:
    """Select rows of every array in ``data`` whose leading axis matches ``mask``.

    Keys that are not per-observation pass through untouched rather than raising, so this works on
    the target dict and the posterior-sample dict without knowing either key set.
    """
    n = len(mask)
    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        out[key] = arr[mask] if arr.ndim >= 1 and arr.shape[0] == n else value
    return out


def _per_branch_figures(requested, posterior, targets, param_names, run_dir, keys) -> None:
    """Write the truth-aware figures once per combination of the 0/1 ``keys``.

    Needed whenever the estimator is conditioned on discrete indicators: it is then a *separate*
    conditional posterior per combination, so a pooled recovery panel superimposes all of them.
    A parameter that is unidentified on one branch by construction (a cavity radius on a disk with
    no cavity) then looks broken in the pooled figure and correct in none. The pooled figures are
    still written -- they are the run-level summary the tuning objectives come from.
    """
    import itertools

    missing = [k for k in keys if k not in targets]
    if not keys or missing:
        log.info("No per-branch split: targets lack %s", missing or "any discrete condition")
        return

    columns = [np.asarray(targets[k]).reshape(-1) for k in keys]
    for combo in itertools.product((0, 1), repeat=len(keys)):
        mask = np.ones(len(columns[0]), dtype=bool)
        for col, want in zip(columns, combo):
            mask &= np.round(col).astype(int) == want
        n_rows = int(mask.sum())
        label = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
        if n_rows < MIN_ROWS_PER_BRANCH:
            log.info("Skipping branch %s: %d rows (< %d)", label, n_rows, MIN_ROWS_PER_BRANCH)
            continue
        suffix = "".join(f"_{k}{v}" for k, v in zip(keys, combo))
        _truth_figures(requested, _mask_rows(posterior, mask), _mask_rows(targets, mask),
                       param_names, run_dir, suffix=suffix)
        log.info("Wrote branch diagnostics for %s (n=%d) -> *%s.png", label, n_rows, suffix)


PRIOR_BOUNDS_NAME = "prior_bounds.npz"


def save_prior_bounds(data: dict, param_names, run_dir: str) -> None:
    """Per-target (min, max) of the training set -- the prior box the corner plots shade."""
    try:
        bounds = {p: np.array([np.min(data[p]), np.max(data[p])], dtype="float64")
                  for p in param_names if p in data}
        np.savez(os.path.join(run_dir, PRIOR_BOUNDS_NAME), **bounds)
    except Exception as exc:  # never abort a training run over a plotting aid
        log.warning("Could not save prior bounds: %s", exc)


def load_prior_bounds(model_dir: str, param_names) -> dict | None:
    """``{name: (lo, hi)}`` written by train, or None if this run predates it."""
    path = os.path.join(model_dir, PRIOR_BOUNDS_NAME)
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return {p: tuple(float(x) for x in z[p]) for p in param_names if p in z}


def save_posterior_plot(posterior, param_names, run_dir: str, truth=None, bounds=None,
                        max_obs: int | None = None) -> None:
    """One corner plot per observation: prior box in the background, median (+ truth) marked.

    ``bounds`` is ``{name: (lo, hi)}`` from ``load_prior_bounds``; the axes are set to the prior
    span so "the posterior fills the prior" is readable at a glance. ``truth`` is ``(n_obs,
    n_params)`` on a simulated test set and None on real data.
    """
    import corner
    import matplotlib.pyplot as plt

    chains = np.stack([np.asarray(posterior[p]).reshape(len(posterior[p]), -1)
                       for p in param_names], axis=-1)          # (n_obs, n_samples, n_params)
    n_obs = chains.shape[0] if max_obs is None else min(chains.shape[0], max_obs)
    lo_hi = [bounds.get(p) if bounds else None for p in param_names]
    # Pad by 5% so the prior box's edges stay visible; 1.0 = corner's own full range for that column.
    ranges = [1.0 if b is None else (b[0] - 0.05 * (b[1] - b[0]), b[1] + 0.05 * (b[1] - b[0]))
              for b in lo_hi]

    for i in range(n_obs):
        s = chains[i]
        med = np.median(s, axis=0)
        fig = corner.corner(
            s, labels=param_names, range=ranges,
            show_titles=True, title_fmt=".3g", quantiles=[0.16, 0.5, 0.84],
            plot_datapoints=False, fill_contours=True, levels=(0.68, 0.95),
            truths=None if truth is None else np.asarray(truth)[i],
            truth_color="tab:red",
        )
        # Prior box behind everything: a band on the diagonals, a rectangle off it.
        axes = np.array(fig.axes).reshape(len(param_names), len(param_names))
        for r in range(len(param_names)):
            for c in range(r + 1):
                ax = axes[r, c]
                if lo_hi[c]:
                    ax.axvspan(*lo_hi[c], color="0.90", zorder=-10)
                if r != c and lo_hi[r]:
                    ax.axhspan(*lo_hi[r], color="0.90", zorder=-10)
        corner.overplot_lines(fig, med, color="tab:blue", linestyle="--", linewidth=1.0)
        corner.overplot_points(fig, med[None], marker="s", color="tab:blue")
        suffix = "" if chains.shape[0] == 1 else f"_obs{i}"
        fig.savefig(os.path.join(run_dir, f"posterior_corner{suffix}.png"),
                    bbox_inches="tight", dpi=120)
        plt.close(fig)
    log.info("Wrote %d posterior corner plot(s) to %s", n_obs, run_dir)
