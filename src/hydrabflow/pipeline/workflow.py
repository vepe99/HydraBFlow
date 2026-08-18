"""Assemble the ``bf.BasicWorkflow`` (adapter + summary network + inference network) from config."""

from __future__ import annotations

from typing import Any

from hydrabflow.pipeline.adapter import build_adapter
from hydrabflow.registry import build_inference_network, build_summary_network

#: Basename of the best-weights checkpoint BayesFlow writes during training.
BEST_WEIGHTS_NAME = "approximator_best"


def build_workflow(cfg, run_dir: str | None = None) -> Any:
    """Build a ``bf.BasicWorkflow`` from the root ``cfg``.

    ``run_dir`` (passed by train and tune) turns on BayesFlow's best-weights checkpointing, so a
    late divergence cannot destroy a converged model; the caller restores those weights before
    saving. Evaluation callers pass nothing and are unaffected.
    """
    import bayesflow as bf
    from omegaconf import OmegaConf

    kwargs = dict(
        adapter=build_adapter(cfg.adapter),
        summary_network=build_summary_network(
            cfg.model.summary_network, cfg.adapter.summary_variables
        ),
        inference_network=build_inference_network(cfg.model.inference_network),
        standardize=list(OmegaConf.to_container(cfg.training.standardize, resolve=True)),
        initial_learning_rate=float(cfg.training.learning_rate),
    )
    if run_dir is not None and bool(cfg.training.save_best_weights):
        kwargs.update(
            checkpoint_filepath=run_dir,
            checkpoint_name=BEST_WEIGHTS_NAME,
            save_best_only=True,
            save_weights_only=True,
        )
    return bf.BasicWorkflow(**kwargs)
