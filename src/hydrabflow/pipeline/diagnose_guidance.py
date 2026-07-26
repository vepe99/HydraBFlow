"""Diagnostic stage: how does the guidance term compare with the diffusion model's own prediction?

The guided sampler runs inside ``jax.lax`` loops, so nothing can be recorded from within it — ``t``
and every intermediate are tracers. This stage therefore **unrolls the reverse ODE eagerly** in
Python, calling the public ``network.score`` / ``network.guidance_update`` at each step so every
quantity is concrete and can be logged.

It is the fastest way to choose ``t_on``, ``guidance_strength``, ``scaling`` and ``max_grad_norm``:
one run costs a fraction of a full evaluation and shows directly whether guidance is doing anything
at all in the gated window.

Per step it records
-------------------
* ``score_norm`` — ``|score|``, the model's own prediction. Grows like ``1/sigma_t`` as ``t -> 0``.
* ``grad_norm`` — ``|grad log p(x_obs | x_hat_0)|``, the raw simulator gradient.
* ``update_norm`` — ``|w(t) * s * scale(t) * grad|``, what is actually added to the score.
* ``ratio`` — ``update_norm / score_norm``. **The headline number.** If this is ~1e-3 in the gated
  window, guidance cannot matter no matter how good its direction is.
* ``cosine`` — ``cos(update, score)``: does guidance agree with the model, or fight it?
* ``dx0_rel`` — ``|sigma_t**2 * update / alpha_t| / |x_hat_0|``: the shift guidance induces in
  *denoised parameter* space, which is the interpretable units.
* ``x0_err`` — ``|x_hat_0 - theta_true|``, so one can see when the denoised estimate becomes
  trustworthy enough for the likelihood to be worth evaluating.
* ``nonfinite_frac`` / ``clipped_frac`` — health of the gradient (bad gradients pitfall).

Both an unguided and a guided trajectory are integrated from the **same base noise**, so the endpoint
difference is attributable to guidance rather than Monte-Carlo noise.

Usage::

    scripts/diagnose_guidance.py model_dir=outputs/lotka_volterra/lv_guided/<timestamp> \\
        simulator=lotka_volterra model=lv_guided training=lv preprocessing=lv
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from hydrabflow.pipeline import io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.checkpoint import load_approximator
from hydrabflow.pipeline.guidance import (
    apply_guidance_overrides,
    attach_guidance,
    detach_guidance,
    get_guided_network,
    resolve_conditions,
)
from hydrabflow.pipeline.workflow import build_workflow
from hydrabflow.preprocessing.registry import build_pipeline
from hydrabflow.simulators.registry import get_simulator
from hydrabflow.utils.logging import get_logger
from hydrabflow.utils.paths import PREPROCESSING_STATE, get_run_dir
from hydrabflow.utils.seed import seed_everything

log = get_logger(__name__)

GUIDANCE_TRACE = "guidance_trace.npz"
GUIDANCE_TRACE_PLOT = "guidance_trace.png"
GUIDANCE_SUMMARY = "guidance_summary.json"


def _norm(x):
    import jax.numpy as jnp

    return jnp.linalg.norm(x, axis=-1, keepdims=True)


def _cos(a, b):
    import jax.numpy as jnp

    return jnp.sum(a * b, axis=-1, keepdims=True) / (_norm(a) * _norm(b) + 1e-30)


def _particle_diagnostics(net, x_hat0, time, reduced_grad, theta_true) -> Dict[str, float]:
    """Whether particle guidance actually differs from point guidance **on real trajectories**.

    The offline study that motivated median-particle guidance perturbed theta isotropically around a
    synthetic point. This records the same quantities against the true denoising spread at the actual
    integration time, which is the check that the offline result transferred:

    * ``width`` — the ``sigma_t/alpha_t`` spread actually used, so the offline blob-width sweep can be
      mapped onto real trajectories;
    * ``cos_point_reduced`` — agreement between the point gradient and the reduced one. Offline this
      fell from 0.997 to 0.000 as the spread grew; if it stays ~1 here, particles change nothing;
    * ``particle_spread`` — mean pairwise cosine between per-particle gradients (how heavy-tailed the
      cloud is; low values are what makes the mean cancel and the median win);
    * ``cos_point_truth`` / ``cos_reduced_truth`` — the direct "is guidance pointing at the answer"
      metric, ``cos(g, theta_true - x_hat_0)``. The reduced one exceeding the point one is the
      success criterion.
    """
    import jax.numpy as jnp

    out: Dict[str, float] = {}
    out["width"] = float(jnp.mean(net.particle_width_at(time)))

    if theta_true is not None:
        to_truth = jnp.asarray(theta_true, dtype=x_hat0.dtype) - x_hat0
        out["cos_reduced_truth"] = float(jnp.mean(_cos(reduced_grad, to_truth)))

    if net.guidance_particles <= 1 or net.guidance_reduce == "point":
        return out

    point_grad = jnp.nan_to_num(net._raw_gradient(x_hat0), nan=0.0, posinf=0.0, neginf=0.0)  # noqa: SLF001
    out["cos_point_reduced"] = float(jnp.mean(_cos(point_grad, reduced_grad)))
    if theta_true is not None:
        to_truth = jnp.asarray(theta_true, dtype=x_hat0.dtype) - x_hat0
        out["cos_point_truth"] = float(jnp.mean(_cos(point_grad, to_truth)))

    grads = jnp.nan_to_num(
        net.particle_gradients(x_hat0, time), nan=0.0, posinf=0.0, neginf=0.0
    )  # (K, batch, dim)
    unit = grads / (_norm(grads) + 1e-30)
    # Mean pairwise cosine == (|mean unit vector|^2 * K - 1) / (K - 1); computed via the Gram trick to
    # avoid materializing a K x K matrix per batch row.
    num_particles = grads.shape[0]
    mean_unit = jnp.mean(unit, axis=0)
    mean_sq = jnp.sum(jnp.square(mean_unit), axis=-1)
    out["particle_spread"] = float(
        jnp.mean((mean_sq * num_particles - 1.0) / max(num_particles - 1, 1))
    )
    return out


def unroll_reverse_ode(
    net,
    z0,
    conditions,
    steps: int,
    guided: bool,
    theta_true: np.ndarray | None = None,
    apply_guidance: bool = True,
) -> tuple[Any, List[Dict[str, float]]]:
    """Integrate ``dz = (f - 0.5 g^2 score) dt`` from ``t=1`` to ``t=0`` with explicit Euler, eagerly.

    Deliberately a plain Euler scheme rather than the production RK45: the point is observability,
    not accuracy, and every stage of a Runge-Kutta step would triple the bookkeeping for no
    diagnostic gain. Endpoint values will differ slightly from a production sample.

    ``apply_guidance=False`` with ``guided=True`` records what guidance *would* do at each step but
    integrates the **unguided** trajectory. That counterfactual mode is the one to trust for judging
    the guidance *direction*: with guidance applied, an unstable run drives the state far out of prior
    within a few steps, the parameter soft-clip saturates, and every subsequent gradient reads as
    zero — so the diagnostics end up describing a diverged trajectory rather than the guidance itself.

    Returns ``(z_final, records)``. ``records`` is empty when ``guided`` is False except for the
    score/x0 columns, so the two runs can be compared column-wise.
    """
    import jax.numpy as jnp

    schedule = net.noise_schedule
    z = z0
    records: List[Dict[str, float]] = []
    dt = -1.0 / steps  # t runs 1 -> 0

    target = net._guidance_target  # noqa: SLF001 - deliberate: we toggle it per step below

    for i in range(steps):
        t = 1.0 + i * dt
        time = jnp.full((jnp.shape(z)[0], 1), t, dtype=z.dtype)

        # --- the model's own prediction, with guidance temporarily off ---
        net.set_guidance_target(None)
        score_plain = net.score(z, time=time, conditions=conditions, training=False)
        net.set_guidance_target(target)

        log_snr = schedule.get_log_snr(t=time, training=False)
        alpha_t, sigma_t = schedule.get_alpha_sigma(log_snr_t=log_snr)
        x_hat0 = (z + jnp.square(sigma_t) * score_plain) / alpha_t

        row: Dict[str, float] = {
            "t": float(t),
            "alpha_t": float(jnp.mean(alpha_t)),
            "sigma_t": float(jnp.mean(sigma_t)),
            "score_norm": float(jnp.mean(_norm(score_plain))),
            "x0_norm": float(jnp.mean(_norm(x_hat0))),
        }
        if theta_true is not None:
            row["x0_err"] = float(jnp.mean(_norm(x_hat0 - jnp.asarray(theta_true, dtype=z.dtype))))

        score = score_plain
        if guided and target is not None:
            raw_grad = net.guidance_gradient(x_hat0, time)
            update = net.guidance_update(x_hat0, time, score_plain)
            if apply_guidance:
                score = score_plain + update

            update_norm = _norm(update)
            row.update(
                grad_norm=float(jnp.mean(_norm(raw_grad))),
                update_norm=float(jnp.mean(update_norm)),
                ratio=float(jnp.mean(update_norm / (_norm(score_plain) + 1e-30))),
                cosine=float(
                    jnp.mean(
                        jnp.sum(update * score_plain, axis=-1, keepdims=True)
                        / (update_norm * _norm(score_plain) + 1e-30)
                    )
                ),
                # Shift induced in denoised-parameter space: dx0 = sigma^2 * dscore / alpha.
                dx0_rel=float(
                    jnp.mean(jnp.square(sigma_t) * update_norm / alpha_t / (_norm(x_hat0) + 1e-30))
                ),
                nonfinite_frac=float(jnp.mean(~jnp.isfinite(raw_grad))),
                clipped_frac=float(jnp.mean(_norm(raw_grad) >= net.max_grad_norm * 0.999)),
            )
            row.update(_particle_diagnostics(net, x_hat0, time, raw_grad, theta_true))

        records.append(row)

        f, g_squared = schedule.get_drift(log_snr_t=log_snr, x=z, training=False)
        z = z + (f - 0.5 * g_squared * score) * dt

    return z, records


def run_diagnose_guidance(cfg):
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    if not cfg.model_dir:
        raise ValueError(
            "diagnose_guidance requires `model_dir` to point at a completed training run, e.g. "
            "model_dir=outputs/<sim>/<model>/<timestamp>"
        )
    model_dir = cfg.model_dir

    import jax.numpy as jnp

    # 1. Rebuild the workflow and load the trained weights (this REPLACES the approximator, so any
    #    guidance target must be attached afterwards).
    workflow = build_workflow(cfg)
    workflow.approximator = load_approximator(model_dir)
    net = get_guided_network(workflow)
    settings = apply_guidance_overrides(net, getattr(cfg.eval, "guidance", None))
    log.info("Guidance settings: %s", settings)

    # 2. One test observation, in both spaces: raw (for the likelihood) and preprocessed (for the
    #    network's conditions).
    simulator = get_simulator(cfg.simulator)
    test_path = os.path.join(cfg.data.data_dir, cfg.eval.test_dataset_name)
    raw = io.load_dataset(test_path)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(model_dir, PREPROCESSING_STATE))
    preprocessed = pipeline.transform(raw)

    index = int(getattr(cfg.eval, "diagnose_index", 0))
    param_names = list(cfg.adapter.inference_variables)
    theta_true = np.concatenate([np.asarray(raw[k])[index].reshape(-1) for k in param_names])
    observable = simulator.observable_keys[0]
    x_obs_raw = np.asarray(raw[observable])[index]

    summary_variables = list(cfg.adapter.summary_variables)
    conditions_row = (
        {observable: np.asarray(preprocessed[observable])[index : index + 1]}
        if summary_variables
        else None
    )

    num_samples = int(getattr(cfg.eval, "diagnose_num_samples", 256))
    steps = int(getattr(cfg.eval, "diagnose_steps", 200))
    conditions = resolve_conditions(workflow.approximator, conditions_row, num_samples)

    # 3. Same base noise for both trajectories, so the endpoint difference is attributable.
    #    An int seed (not a raw PRNGKey) — BayesFlow's resolve_seed does a truthiness check on it.
    z0 = net.base_distribution.sample((num_samples,), seed=int(cfg.seed))

    attach_guidance(workflow, simulator, x_obs_raw, net=net)
    log.info("Unrolling %d Euler steps, %d particles (unguided) ...", steps, num_samples)
    detach_guidance(net)
    z_unguided, rec_unguided = unroll_reverse_ode(
        net, z0, conditions, steps, guided=False, theta_true=theta_true
    )

    attach_guidance(workflow, simulator, x_obs_raw, net=net)
    log.info("Unrolling %d Euler steps, %d particles (guided) ...", steps, num_samples)
    z_guided, rec_guided = unroll_reverse_ode(
        net, z0, conditions, steps, guided=True, theta_true=theta_true
    )

    # Counterfactual pass: what guidance WOULD do along the unguided trajectory. Direction quality
    # has to be read off this one — see unroll_reverse_ode's docstring on why applying guidance
    # contaminates its own diagnostics once the run destabilizes.
    log.info("Unrolling %d Euler steps, %d particles (counterfactual) ...", steps, num_samples)
    _, rec_counterfactual = unroll_reverse_ode(
        net, z0, conditions, steps, guided=True, theta_true=theta_true, apply_guidance=False
    )

    # 4. Persist the trace + a summary restricted to the gated window (where guidance is live).
    trace = {
        key: np.asarray([row.get(key, np.nan) for row in rec_guided], dtype=np.float64)
        for key in sorted({k for row in rec_guided for k in row})
    }
    trace["x0_err_unguided"] = np.asarray(
        [row.get("x0_err", np.nan) for row in rec_unguided], dtype=np.float64
    )
    # Counterfactual columns (cf_*): guidance measured along the UNGUIDED path. These are the ones to
    # judge the guidance direction by; the un-prefixed ones describe the guided run, warts and all.
    for key in sorted({k for row in rec_counterfactual for k in row}):
        trace[f"cf_{key}"] = np.asarray(
            [row.get(key, np.nan) for row in rec_counterfactual], dtype=np.float64
        )
    np.savez(os.path.join(run_dir, GUIDANCE_TRACE), **trace)

    active = trace["t"] <= net.t_on
    summary = {
        "settings": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in settings.items()},
        "model_dir": model_dir,
        "test_index": index,
        "num_particles": num_samples,
        "euler_steps": steps,
        "theta_true": theta_true.tolist(),
        "gated_window": {
            "n_steps_active": int(active.sum()),
            "fraction_of_trajectory": float(active.mean()),
        },
    }
    for key in (
        "ratio", "cosine", "grad_norm", "score_norm", "update_norm", "dx0_rel",
        "width", "cos_point_reduced", "particle_spread", "cos_point_truth", "cos_reduced_truth",
    ):
        values = trace.get(key)
        if values is None:
            continue
        window = values[active]
        window = window[np.isfinite(window)]
        if window.size:
            summary[key] = {
                "median_in_window": float(np.median(window)),
                "max_in_window": float(np.max(window)),
                "min_in_window": float(np.min(window)),
            }
    for key in ("nonfinite_frac", "clipped_frac"):
        values = trace.get(key)
        if values is not None:
            window = values[active]
            summary[key] = {"max_in_window": float(np.nanmax(window)) if window.size else 0.0}

    shift = np.asarray(jnp.mean(jnp.abs(z_guided - z_unguided), axis=0))
    summary["mean_abs_endpoint_shift"] = dict(zip(param_names, shift.tolist()))
    summary["endpoint_error"] = {
        "unguided": float(np.mean(np.abs(np.asarray(z_unguided) - theta_true))),
        "guided": float(np.mean(np.abs(np.asarray(z_guided) - theta_true))),
    }

    with open(os.path.join(run_dir, GUIDANCE_SUMMARY), "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Guidance/score norm ratio in window: %s", summary.get("ratio"))
    log.info("Endpoint |error| unguided vs guided: %s", summary["endpoint_error"])
    if "ratio" in summary and summary["ratio"]["max_in_window"] < 1e-2:
        log.warning(
            "Guidance is <1%% of the score everywhere in the gated window, so it cannot "
            "materially change the samples. Increase guidance_strength, use scaling=snr or "
            "norm_matched, or widen the window (raise t_on)."
        )

    _plot_trace(trace, net, run_dir)
    log.info("Guidance diagnostics complete. Artifacts in %s", run_dir)
    return summary


def _plot_trace(trace, net, run_dir: str) -> None:
    """Four-panel diagnostic figure; best-effort (never aborts the stage)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = trace["t"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

        ax = axes[0, 0]
        ax.plot(t, trace["score_norm"], label="|score| (model)")
        if "grad_norm" in trace:
            ax.plot(t, trace["grad_norm"], label="|grad log p| (raw)")
        if "update_norm" in trace:
            ax.plot(t, trace["update_norm"], label="|update| (applied)")
        ax.set_yscale("log")
        ax.set_ylabel("norm")
        ax.set_title("Magnitudes")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        if "ratio" in trace:
            ax.plot(t, trace["ratio"], color="C3")
        ax.axhline(1.0, ls=":", c="k", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylabel("|update| / |score|")
        ax.set_title("Guidance vs model prediction")

        ax = axes[1, 0]
        if "cosine" in trace:
            ax.plot(t, trace["cosine"], color="C2", label="vs score")
        # The success criterion for particle guidance: does the reduced direction point at the truth
        # better than the point (Tweedie) direction does?
        for key, style, label in (
            ("cos_point_truth", "--", "point vs truth"),
            ("cos_reduced_truth", "-", "reduced vs truth"),
            ("cos_point_reduced", ":", "point vs reduced"),
        ):
            if key in trace and np.isfinite(trace[key]).any():
                ax.plot(t, trace[key], style, lw=1.2, label=label)
        ax.axhline(0.0, ls=":", c="k", lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("cosine")
        ax.set_xlabel("diffusion time t  (sampling runs right to left)")
        ax.set_title("Agreement in direction")
        ax.legend(fontsize=7)

        ax = axes[1, 1]
        if "x0_err" in trace:
            ax.plot(t, trace["x0_err"], label="guided")
        if "x0_err_unguided" in trace:
            ax.plot(t, trace["x0_err_unguided"], ls="--", label="unguided")
        ax.set_yscale("log")
        ax.set_ylabel(r"$|\hat{x}_0 - \theta_{true}|$")
        ax.set_xlabel("diffusion time t  (sampling runs right to left)")
        ax.set_title("Denoised-estimate error")
        ax.legend(fontsize=8)

        for ax in axes.ravel():
            ax.axvspan(0.0, net.t_on, color="k", alpha=0.06)
            ax.invert_xaxis()  # follow the direction of sampling
        fig.suptitle(
            f"Guidance trace — strength={net.guidance_strength}, t_on={net.t_on}, "
            f"scaling={net.scaling} (shaded = guidance active)"
        )
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, GUIDANCE_TRACE_PLOT), dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # plotting must never abort the diagnostic
        log.warning("Could not plot guidance trace: %s", exc)


cli = make_cli(run_diagnose_guidance)


if __name__ == "__main__":
    cli()
