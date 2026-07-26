"""Stage 5: application to real (observed) data.

Like :mod:`evaluate`, but the input is a user-provided real-data ``.npz`` with no ground-truth
parameters: there is no prior sampling and no resimulation. We replay the fitted preprocessing,
draw posterior samples, and write truth-free diagnostics (posterior pair/marginal plots).
Generalizes the reference ``main_eval_gaiastreams_*`` scripts.

Real data often needs a couple of dataset-specific touch-ups (e.g. overriding synthetic
measurement errors with the instrument's real errors). Do those by adding preprocessing steps or
augmentations restricted to this stage — the hook is the ``cfg.augmentation``/``cfg.preprocessing``
groups, which you can override on the CLI for this run.
"""

from __future__ import annotations

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


def run_real_evaluation(cfg):
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    if not cfg.model_dir:
        raise ValueError("evaluate_real requires `model_dir` (a completed training run).")
    if not cfg.data.real_data_path:
        raise ValueError("Set `data.real_data_path` to your observed-data .npz.")

    # 1. Load model + fitted preprocessing.
    workflow = build_workflow(cfg)
    workflow.approximator = load_approximator(cfg.model_dir)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(cfg.model_dir, PREPROCESSING_STATE))

    # 2. Load + preprocess the real observation(s). No ground truth present.
    real_data = io.load_dataset(cfg.data.real_data_path)
    real_data = pipeline.transform(real_data)

    # 3. Sample the posterior and persist it. `inference.sample_kwargs` is forwarded verbatim so
    #    sampler options (integrator method / steps) are configurable without editing this stage.
    from hydrabflow.pipeline.evaluate import _sample_kwargs

    posterior = workflow.sample(
        num_samples=int(cfg.inference.num_samples),
        conditions=real_data,
        batch_size=int(cfg.inference.batch_size),
        **_sample_kwargs(cfg.inference),
    )
    np.savez(
        os.path.join(run_dir, POSTERIOR_SAMPLES),
        **{k: np.asarray(v) for k, v in posterior.items()},
    )

    # 4. Truth-free diagnostics: posterior marginals / pair plot.
    _save_posterior_plot(posterior, list(cfg.adapter.inference_variables), run_dir)
    log.info("Real-data inference complete. Artifacts in %s", run_dir)
    return posterior


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
