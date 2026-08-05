"""Stage 3: posterior inference on held-out data, simulated or real.

Loads the approximator + fitted preprocessing from ``cfg.model_dir``, then samples. Two modes, one
code path: the simulated test set (has ground truth -> truth-aware diagnostics), or
``data.real_data_path=<your.npz>`` (no truth, no resimulation -> posterior pair plots only).
"""

from __future__ import annotations

import logging
import os

from hydrabflow.pipeline import artifacts, io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.workflow import build_workflow
from hydrabflow.registry import build_pipeline
from hydrabflow.utils.paths import PREPROCESSING_STATE, get_run_dir
from hydrabflow.utils.seed import seed_everything

log = logging.getLogger(__name__)


def run_evaluation(cfg):
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    if not cfg.model_dir:
        raise ValueError(
            "evaluate requires `model_dir` to point at a completed training run, e.g. "
            "model_dir=outputs/<sim>/<run>/<timestamp>"
        )

    # 1. Rebuild the workflow and load the trained approximator into it.
    workflow = build_workflow(cfg)
    workflow.approximator = artifacts.load_approximator(cfg.model_dir)

    # 2. Load the data and replay the *fitted* preprocessing (no re-fit, no split).
    real = bool(cfg.data.real_data_path)
    path = cfg.data.real_data_path or os.path.join(cfg.data.data_dir, cfg.eval.test_dataset_name)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(cfg.model_dir, PREPROCESSING_STATE))
    data = pipeline.transform(io.load_dataset(path))

    # 3. Sample the posterior for every observation, then score it if the truth is known.
    posterior = workflow.sample(
        num_samples=int(cfg.eval.num_samples),
        conditions=data,
        batch_size=int(cfg.eval.batch_size),
    )
    param_names = list(cfg.adapter.inference_variables)
    artifacts.save_posterior(posterior, run_dir)
    if real:
        artifacts.save_posterior_plot(posterior, param_names, run_dir)
    else:
        artifacts.run_diagnostics(cfg, posterior, data, param_names, run_dir)

    log.info("Inference on %s complete. Artifacts in %s", path, run_dir)
    return posterior


cli = make_cli(run_evaluation)


if __name__ == "__main__":
    cli()
