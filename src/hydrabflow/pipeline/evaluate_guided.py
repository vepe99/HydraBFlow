"""Guided evaluation: does simulator-gradient guidance improve the posterior, and at what cost?

Runs **paired** guided / unguided posterior sampling on the same held-out observations from the same
trained network, then reports RMSE and calibration side by side. Both numbers matter and they can
move in opposite directions:

* Arm B (conditional network, ``model=lv_guided``): the network already encodes ``p(theta | x_obs)``,
  so adding ``grad log p(x_obs | theta)`` targets ``p(theta|x_obs) * p(x_obs|theta)^w`` —
  over-concentrated *by construction*. RMSE may fall while calibration degrades. That trade-off is
  the measurement, not a bug.
* Arm A (unconditional network, ``model=lv_prior_only``): the network encodes only the prior, so all
  data information arrives through guidance. Here calibration is a genuine test of the method.

Why one observation at a time
-----------------------------
The guided sampler is driven per observation. BayesFlow's sampler slices and repeats array kwargs to
align with ``batch * num_samples`` rows (``approximators/helpers/samplers.py``), and getting a
per-row observation aligned through that is easy to get subtly wrong. Sampling one observation at a
time makes every particle in a call share the same ``x_obs``, so the guidance likelihood needs no
alignment logic at all. It also means guided and unguided draws for an observation come from the same
network state, which is what makes the comparison paired.

Outputs
-------
``posterior_guided.npz``, ``posterior_unguided.npz``, ``metrics_guided.json`` (per-arm RMSE,
calibration error, mean posterior std, plus the guidance settings in force), calibration/recovery
plots for both arms, and — when ``eval.n_reference_obs > 0`` — MCMC reference comparisons.

The reference sampler is **off by default** and not yet validated; see ``pipeline.reference``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from hydrabflow.pipeline import io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.checkpoint import load_approximator
from hydrabflow.pipeline.evaluate import _sample_kwargs
from hydrabflow.pipeline.guidance import (
    apply_guidance_overrides,
    attach_guidance,
    detach_guidance,
    get_guided_network,
)
from hydrabflow.pipeline.workflow import build_workflow
from hydrabflow.preprocessing.registry import build_pipeline
from hydrabflow.simulators.registry import get_simulator
from hydrabflow.utils.logging import get_logger
from hydrabflow.utils.paths import PREPROCESSING_STATE, get_run_dir
from hydrabflow.utils.seed import seed_everything

log = get_logger(__name__)

POSTERIOR_GUIDED = "posterior_guided.npz"
POSTERIOR_UNGUIDED = "posterior_unguided.npz"
REFERENCE_POSTERIOR = "posterior_reference.npz"
METRICS_GUIDED = "metrics_guided.json"


def run_guided_evaluation(cfg):
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    if not cfg.model_dir:
        raise ValueError(
            "evaluate_guided requires `model_dir` to point at a completed training run, e.g. "
            "model_dir=outputs/<sim>/<model>/<timestamp>"
        )
    model_dir = cfg.model_dir

    # 1. Load the trained approximator (this REPLACES the config-built one, so guidance must be
    #    attached afterwards — see pipeline.guidance).
    workflow = build_workflow(cfg)
    workflow.approximator = load_approximator(model_dir)
    net = get_guided_network(workflow)
    settings = apply_guidance_overrides(net, cfg.eval.guidance)
    log.info("Guidance settings in force: %s", settings)

    simulator = get_simulator(cfg.simulator)
    param_names = list(cfg.adapter.inference_variables)
    summary_variables = list(cfg.adapter.summary_variables)
    observable = simulator.observable_keys[0]

    # 2. Test set in both spaces: raw (physical, for the likelihood) and preprocessed (for the
    #    network's conditions). Using the preprocessed copy for the likelihood would be a silent bug.
    test_path = os.path.join(cfg.data.data_dir, cfg.eval.test_dataset_name)
    raw = io.load_dataset(test_path)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(model_dir, PREPROCESSING_STATE))
    preprocessed = pipeline.transform(raw)

    n_available = len(np.asarray(raw[param_names[0]]))
    n_obs = min(int(cfg.eval.n_guided_obs), n_available)
    if n_obs < int(cfg.eval.n_guided_obs):
        log.warning(
            "eval.n_guided_obs=%d exceeds the test set size (%d); using %d.",
            int(cfg.eval.n_guided_obs), n_available, n_obs,
        )
    num_samples = int(cfg.eval.num_samples)
    sample_kwargs = _sample_kwargs(cfg.eval)

    truth = np.stack(
        [np.concatenate([np.asarray(raw[k])[i].reshape(-1) for k in param_names]) for i in range(n_obs)]
    )  # (n_obs, dim)

    guided: List[np.ndarray] = []
    unguided: List[np.ndarray] = []

    log.info("Sampling %d observations x %d draws, guided and unguided ...", n_obs, num_samples)
    for i in range(n_obs):
        conditions = (
            {observable: np.asarray(preprocessed[observable])[i : i + 1]}
            if summary_variables
            else None
        )
        seed_i = int(cfg.seed) + i  # same seed for both arms => paired comparison

        detach_guidance(net)
        unguided.append(
            _draw(workflow, conditions, num_samples, param_names, seed_i, sample_kwargs)
        )

        attach_guidance(workflow, simulator, np.asarray(raw[observable])[i], net=net)
        guided.append(
            _draw(workflow, conditions, num_samples, param_names, seed_i, sample_kwargs)
        )

        if (i + 1) % max(1, n_obs // 10) == 0:
            log.info("  %d/%d observations done", i + 1, n_obs)
    detach_guidance(net)

    guided_arr = np.stack(guided)      # (n_obs, num_samples, dim)
    unguided_arr = np.stack(unguided)

    _save(os.path.join(run_dir, POSTERIOR_GUIDED), guided_arr, param_names, truth)
    _save(os.path.join(run_dir, POSTERIOR_UNGUIDED), unguided_arr, param_names, truth)

    # 3. Metrics for both arms, computed identically.
    metrics: Dict[str, Any] = {
        "model_dir": model_dir,
        "guidance_settings": {k: _jsonable(v) for k, v in settings.items()},
        "n_obs": n_obs,
        "num_samples": num_samples,
        "parameters": param_names,
        "arms": {
            "unguided": _arm_metrics(unguided_arr, truth, param_names),
            "guided": _arm_metrics(guided_arr, truth, param_names),
        },
    }
    metrics["delta"] = {
        key: metrics["arms"]["guided"][key] - metrics["arms"]["unguided"][key]
        for key in ("rmse", "calibration_error", "mean_posterior_std", "mean_abs_z")
    }

    # 4. Optional MCMC reference (off by default; sampler not yet validated).
    n_ref = int(cfg.eval.n_reference_obs)
    if n_ref > 0:
        metrics["reference"] = _reference_comparison(
            cfg, simulator, raw, observable, guided_arr, unguided_arr, param_names, n_ref, run_dir
        )

    with open(os.path.join(run_dir, METRICS_GUIDED), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("unguided: %s", metrics["arms"]["unguided"])
    log.info("guided:   %s", metrics["arms"]["guided"])
    log.info("delta (guided - unguided): %s", metrics["delta"])
    if metrics["delta"]["rmse"] < 0 and metrics["delta"]["calibration_error"] > 0:
        log.warning(
            "Guidance improved RMSE but worsened calibration — the expected signature of "
            "likelihood double-counting. Compare against the prior-only arm before concluding "
            "that guidance helps."
        )

    _plot_arms(guided_arr, unguided_arr, truth, param_names, run_dir)
    log.info("Guided evaluation complete. Artifacts in %s", run_dir)
    return metrics


def _draw(workflow, conditions, num_samples, param_names, seed, sample_kwargs) -> np.ndarray:
    """One posterior sample set for a single observation, as ``(num_samples, dim)``."""
    posterior = workflow.sample(
        num_samples=num_samples,
        conditions=conditions,
        seed=seed,
        **sample_kwargs,
    )
    return np.concatenate(
        [np.asarray(posterior[k]).reshape(num_samples, -1) for k in param_names], axis=-1
    )


def _save(path: str, samples: np.ndarray, param_names: List[str], truth: np.ndarray) -> None:
    """Persist per-parameter posterior draws plus the ground truth, in the repo's npz convention."""
    payload = {name: samples[:, :, i] for i, name in enumerate(param_names)}
    payload |= {f"truth_{name}": truth[:, i] for i, name in enumerate(param_names)}
    np.savez(path, **payload)


def _arm_metrics(samples: np.ndarray, truth: np.ndarray, param_names: List[str]) -> Dict[str, Any]:
    """RMSE of the posterior mean, calibration error from rank statistics, and spread.

    ``calibration_error`` is the mean absolute deviation of the empirical rank CDF from uniform
    (0 = perfectly calibrated). It is computed here rather than via ``bf.diagnostics`` so guided and
    unguided arms are scored by identical code on identically shaped arrays.
    """
    mean = samples.mean(axis=1)                       # (n_obs, dim)
    std = samples.std(axis=1) + 1e-12
    rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
    per_param_rmse = np.sqrt(np.mean((mean - truth) ** 2, axis=0))

    # Rank of the truth among the posterior draws, per observation and parameter.
    ranks = (samples < truth[:, None, :]).mean(axis=1)  # (n_obs, dim), uniform if calibrated
    calib = float(np.mean([_uniformity_error(ranks[:, d]) for d in range(ranks.shape[1])]))
    z = (mean - truth) / std

    return {
        "rmse": rmse,
        "per_parameter_rmse": dict(zip(param_names, per_param_rmse.tolist())),
        "calibration_error": calib,
        "per_parameter_calibration_error": dict(
            zip(param_names, [_uniformity_error(ranks[:, d]) for d in range(ranks.shape[1])])
        ),
        "mean_posterior_std": float(np.mean(std)),
        "per_parameter_posterior_std": dict(zip(param_names, std.mean(axis=0).tolist())),
        # |z| >> 1 means the posterior is over-confident (truth outside its own error bars).
        "mean_abs_z": float(np.mean(np.abs(z))),
    }


def _uniformity_error(ranks: np.ndarray) -> float:
    """Mean absolute deviation of the empirical CDF of ``ranks`` from Uniform(0,1)."""
    if ranks.size == 0:
        return float("nan")
    sorted_ranks = np.sort(ranks)
    expected = (np.arange(ranks.size) + 0.5) / ranks.size
    return float(np.mean(np.abs(sorted_ranks - expected)))


def _reference_comparison(
    cfg, simulator, raw, observable, guided_arr, unguided_arr, param_names, n_ref, run_dir
) -> Dict[str, Any]:
    """Compare both arms against MCMC reference posteriors on the first ``n_ref`` observations."""
    from hydrabflow.pipeline import reference

    settings = reference.resolve_settings(cfg.eval.reference)
    out: Dict[str, Any] = {"settings": {k: _jsonable(v) for k, v in settings.items()}, "per_obs": []}
    saved: Dict[str, np.ndarray] = {}

    for i in range(min(n_ref, guided_arr.shape[0])):
        try:
            res = reference.reference_posterior(
                simulator, np.asarray(raw[observable])[i], settings, seed=int(cfg.seed) + i
            )
        except Exception as exc:
            log.warning("reference posterior failed for observation %d: %s", i, exc)
            continue
        ref = res["flat_samples"]
        saved[f"obs{i}"] = ref
        out["per_obs"].append(
            {
                "index": i,
                "r_hat_max": float(np.nanmax(res["r_hat"])),
                "ess_min": float(np.nanmin(res["ess"])),
                "preconditioner": res.get("preconditioner", {}).get("kind"),
                "mmd_guided": _mmd(guided_arr[i], ref),
                "mmd_unguided": _mmd(unguided_arr[i], ref),
                "wasserstein_guided": _wasserstein(guided_arr[i], ref, param_names),
                "wasserstein_unguided": _wasserstein(unguided_arr[i], ref, param_names),
            }
        )
    if saved:
        np.savez(os.path.join(run_dir, REFERENCE_POSTERIOR), **saved)
    return out


def _mmd(a: np.ndarray, b: np.ndarray, max_n: int = 1000) -> float:
    """Unbiased squared MMD with an RBF kernel at the median heuristic bandwidth."""
    rng = np.random.default_rng(0)
    a = a[rng.choice(len(a), min(len(a), max_n), replace=False)]
    b = b[rng.choice(len(b), min(len(b), max_n), replace=False)]
    pooled = np.vstack([a, b])
    d2 = np.sum((pooled[:, None, :] - pooled[None, :, :]) ** 2, axis=-1)
    median = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    k = np.exp(-d2 / median)
    n, m = len(a), len(b)
    kaa = (k[:n, :n].sum() - np.trace(k[:n, :n])) / (n * (n - 1))
    kbb = (k[n:, n:].sum() - np.trace(k[n:, n:])) / (m * (m - 1))
    kab = k[:n, n:].mean()
    return float(kaa + kbb - 2 * kab)


def _wasserstein(a: np.ndarray, b: np.ndarray, param_names: List[str]) -> Dict[str, float]:
    """Per-marginal 1-Wasserstein distance (scipy; avoids a scikit-learn C2ST dependency)."""
    from scipy.stats import wasserstein_distance

    return {
        name: float(wasserstein_distance(a[:, i], b[:, i])) for i, name in enumerate(param_names)
    }


def _jsonable(value):
    return value.item() if isinstance(value, np.generic) else value


def _plot_arms(guided, unguided, truth, param_names, run_dir: str) -> None:
    """Recovery + rank-ECDF panels for both arms; best-effort (never aborts the stage)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dim = len(param_names)
        fig, axes = plt.subplots(2, dim, figsize=(3.2 * dim, 6.4))
        axes = np.atleast_2d(axes)

        for d, name in enumerate(param_names):
            ax = axes[0, d]
            for arr, label, color in ((unguided, "unguided", "C0"), (guided, "guided", "C3")):
                ax.errorbar(
                    truth[:, d], arr[:, :, d].mean(axis=1), yerr=arr[:, :, d].std(axis=1),
                    fmt="o", ms=2.5, lw=0.6, alpha=0.55, color=color, label=label,
                )
            lo, hi = truth[:, d].min(), truth[:, d].max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_title(name)
            ax.set_xlabel("truth")
            if d == 0:
                ax.set_ylabel("posterior mean")
                ax.legend(fontsize=7)

            ax = axes[1, d]
            for arr, label, color in ((unguided, "unguided", "C0"), (guided, "guided", "C3")):
                ranks = np.sort((arr[:, :, d] < truth[:, None, d]).mean(axis=1))
                ax.plot(ranks, np.linspace(0, 1, len(ranks)), color=color, label=label)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("rank of truth")
            if d == 0:
                ax.set_ylabel("ECDF (uniform = calibrated)")

        fig.suptitle("Guided vs unguided: recovery (top) and calibration (bottom)")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "guided_vs_unguided.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        log.warning("Could not plot guided/unguided comparison: %s", exc)


cli = make_cli(run_guided_evaluation)


if __name__ == "__main__":
    cli()
