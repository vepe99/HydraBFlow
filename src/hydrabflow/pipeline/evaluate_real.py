"""Stage 5: application to real (observed) data.

Like :mod:`evaluate`, but the input is a user-provided real-data ``.npz`` with no ground-truth
parameters: there is no prior sampling and no resimulation. We replay the fitted preprocessing,
draw posterior samples, and write truth-free diagnostics (posterior pair plots).

Three paths, selected by ``composition.level``:

* ``none`` — single-level: sample each observation independently (the template default).
* ``global`` — the real dataset is one *group* of exchangeable members (e.g. the observed
  streams); their evidence is pooled with ``compositional_sample`` + the simulator's prior
  score. The saved posterior is the input for the local level. Generalizes
  ``main_eval_gaiastreams_new_rotationcurve_agama.py``.
* ``local`` — per-member local parameters via **ancestral sampling**: global draws come from a
  previously computed global posterior (``composition.global_run_dir`` = that evaluation's run
  dir). Posterior samples are mapped back to physical units through the preprocessing inverse.
  Generalizes ``main_eval_gaiastreams_local_new_jonas_streamnorm_rotationcurve_agama.py``.

Real data usually needs its own (reduced) preprocessing/augmentation presets — the
observations already carry the instrument's selection and noise, so only the representation
steps are replayed (see ``conf/preprocessing/stream_real_*`` and
``conf/augmentation/stream_real_*``).
"""

from __future__ import annotations

import os

import numpy as np

from hydrabflow.pipeline import io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.checkpoint import load_approximator
from hydrabflow.pipeline.compositional import (
    apply_augmentations_once,
    composition_level,
    condition_keys,
    group_members,
    prior_score_from_spec,
)
from hydrabflow.pipeline.workflow import build_workflow
from hydrabflow.preprocessing.registry import build_pipeline
from hydrabflow.utils.logging import get_logger
from hydrabflow.utils.paths import POSTERIOR_SAMPLES, PREPROCESSING_STATE, get_run_dir
from hydrabflow.utils.seed import seed_everything

log = get_logger(__name__)


def run_real_evaluation(cfg):
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    if not cfg.model_dir:
        raise ValueError("evaluate_real requires `model_dir` (a completed training run).")
    if not cfg.data.real_data_path:
        raise ValueError("Set `data.real_data_path` to your observed-data .npz.")

    level = composition_level(cfg)
    if level in ("global", "local"):
        return _evaluate_real_compositional(cfg, level, run_dir)

    # ------------------------------ single-level (template default) ----------------------- #
    workflow = build_workflow(cfg)
    workflow.approximator = load_approximator(cfg.model_dir)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(cfg.model_dir, PREPROCESSING_STATE))

    real_data = io.load_dataset(cfg.data.real_data_path)
    real_data = pipeline.transform(real_data)

    posterior = workflow.sample(
        num_samples=int(cfg.inference.num_samples),
        conditions=real_data,
        batch_size=int(cfg.inference.batch_size),
    )
    _save_posterior(posterior, run_dir)
    _save_posterior_plot(posterior, list(cfg.adapter.inference_variables), run_dir)
    log.info("Real-data inference complete. Artifacts in %s", run_dir)
    return posterior


# ------------------------------------------------------------------------------------------- #
# Compositional real-data evaluation
# ------------------------------------------------------------------------------------------- #


def _max_particles(cfg) -> int | None:
    params = getattr(cfg.augmentation, "params", None)
    if params is None:
        return None
    value = params.get("max_particles") if hasattr(params, "get") else None
    return int(value) if value else None


def _prepare_real_members(cfg):
    """Load the observed group and normalize it to flat member rows.

    Expected layout (as produced for the Gaia streams): the member observable
    ``(1, m, particles, features)``; per-member aux arrays ``(m, ...)``; ``j`` with one entry
    per member. The particle axis is truncated to ``augmentation.params.max_particles`` (the
    training compaction length), and 2-D masks are lifted to ``(m, 1, particles)``.
    """
    data = io.load_dataset(cfg.data.real_data_path)
    m = int(np.asarray(data["j"]).size)

    max_p = _max_particles(cfg)
    flat = {}
    for key, arr in data.items():
        arr = np.asarray(arr)
        if key != "j" and max_p:
            if arr.ndim == 2:
                arr = arr[:, :max_p]
            elif arr.ndim >= 3:
                arr = arr[:, :, :max_p]
        if arr.ndim >= 3 and arr.shape[0] == 1 and arr.shape[1] == m:
            arr = arr.reshape(m, *arr.shape[2:])
        if key in ("attention_mask", "vlos_mask") and arr.ndim == 2:
            arr = arr[:, None, :]
        flat[key] = arr
    flat["j"] = np.asarray(data["j"]).reshape(m, 1).astype(float)
    return flat, m


def _evaluate_real_compositional(cfg, level: str, run_dir: str):
    workflow = build_workflow(cfg)
    workflow.approximator = load_approximator(cfg.model_dir)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(cfg.model_dir, PREPROCESSING_STATE))

    flat, m = _prepare_real_members(cfg)
    flat = pipeline.transform(flat)
    flat = apply_augmentations_once(flat, cfg, pipeline, int(cfg.seed))

    # One observed group: conditions shaped (1, m, ...). Keys the adapter consumes but the
    # observation does not carry (the global parameters, at the local level) are supplied by
    # the sampler itself (ancestral draws), not by the data.
    grouped = group_members(flat, 1, m)
    conditions = {k: grouped[k] for k in condition_keys(cfg) if k in grouped}

    from omegaconf import OmegaConf

    sample_kwargs = cfg.eval.sample_kwargs
    sample_kwargs = (
        OmegaConf.to_container(sample_kwargs, resolve=True)
        if OmegaConf.is_config(sample_kwargs)
        else dict(sample_kwargs)
    )

    if level == "global":
        from hydrabflow.simulators.registry import get_simulator

        prior_score = prior_score_from_spec(get_simulator(cfg.simulator).prior_spec_global)
        log.info("Compositional (global) sampling on the observed group of %d members", m)
        posterior = workflow.compositional_sample(
            num_samples=int(cfg.inference.num_samples),
            conditions=conditions,
            compute_prior_score=prior_score,
            batch_size=int(cfg.inference.batch_size),
            **sample_kwargs,
        )
        _save_posterior(posterior, run_dir)
        _save_posterior_plot(posterior, list(cfg.adapter.inference_variables), run_dir)
        log.info("Global posterior saved. Point composition.global_run_dir here for the "
                 "local level. Artifacts in %s", run_dir)
        return posterior

    # ------------------------------------ local level -------------------------------------- #
    if not cfg.composition.global_run_dir:
        raise ValueError(
            "composition.level=local real-data evaluation needs composition.global_run_dir "
            "(the run dir of the global model's real-data evaluation, holding its posterior)."
        )
    global_posterior = io.load_dataset(
        os.path.join(cfg.composition.global_run_dir, POSTERIOR_SAMPLES)
    )
    ancestral = {}
    for key in cfg.adapter.inference_conditions:
        if key in global_posterior:
            arr = np.asarray(global_posterior[key])
            ancestral[key] = arr.reshape(1, -1, arr.shape[-1] if arr.ndim > 1 else 1)
    if not ancestral:
        raise ValueError(
            f"No global-parameter samples found in {cfg.composition.global_run_dir}: keys "
            f"{list(global_posterior)} do not overlap adapter.inference_conditions."
        )

    log.info("Ancestral (local) sampling: %d members x %d global draws",
             m, next(iter(ancestral.values())).shape[1])
    posterior = workflow.ancestral_sample(
        conditions=conditions,
        ancestral_conditions=ancestral,
        batch_size=int(cfg.inference.batch_size),
        **sample_kwargs,
    )

    # Back to physical units (per-stream prior normalization is invertible given j).
    posterior = dict(posterior)
    posterior["j"] = grouped["j"]
    posterior = pipeline.inverse_transform(posterior)
    posterior.pop("j")
    _save_posterior(posterior, run_dir)

    # Truth-free per-member pair plots: leading axis = member.
    param_names = list(cfg.adapter.inference_variables)
    per_member = {
        k: np.asarray(v)[0].reshape(m, -1, 1) for k, v in posterior.items() if k in param_names
    }
    _save_posterior_plot(per_member, param_names, run_dir)
    log.info("Local (ancestral) real-data inference complete. Artifacts in %s", run_dir)
    return posterior


def _save_posterior(posterior, run_dir: str) -> None:
    np.savez(
        os.path.join(run_dir, POSTERIOR_SAMPLES),
        **{k: np.asarray(v) for k, v in posterior.items()},
    )


def _save_posterior_plot(posterior, param_names, run_dir) -> None:
    """Save a posterior pair plot per observation (real data has no ground truth)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import bayesflow as bf

        fn = getattr(bf.diagnostics, "pairs_posterior", None) or getattr(
            bf.diagnostics, "pairs_samples", None
        )
        if fn is None:
            log.warning("No posterior pair-plot helper found in bayesflow.diagnostics.")
            return
        n_obs = int(np.asarray(next(iter(posterior.values()))).shape[0])
        for i in range(n_obs):
            single = {k: np.asarray(v)[i] for k, v in posterior.items()}  # (n_samples, 1) each
            fig = fn(estimates=single, variable_names=param_names)
            suffix = "" if n_obs == 1 else f"_obs{i}"
            fig.savefig(os.path.join(run_dir, f"posterior_pairs{suffix}.png"), bbox_inches="tight")
    except Exception as exc:
        log.warning("posterior plot failed: %s", exc)


cli = make_cli(run_real_evaluation)


if __name__ == "__main__":
    cli()
