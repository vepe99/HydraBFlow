"""Assemble a single-level BayesFlow workflow from config.

Uses ``bf.BasicWorkflow`` (a ``ContinuousApproximator`` under the hood) — no compositional /
hierarchical machinery. Combines the adapter, the summary network, and the inference network.
"""

from __future__ import annotations

from typing import Any

from hydrabflow.networks.factory import build_inference_network, build_summary_network
from hydrabflow.pipeline.adapter import build_adapter


#: Basename (sans extension) of the best-weights checkpoint BayesFlow writes during training.
BEST_WEIGHTS_NAME = "approximator_best"


def build_workflow(cfg, run_dir: str | None = None) -> Any:
    """Build a ``bf.BasicWorkflow`` from the root ``cfg``.

    ``run_dir`` (set by the train and tune stages) turns on BayesFlow's built-in best-weights
    checkpointing: it writes ``<run_dir>/approximator_best.weights.h5`` whenever the monitored
    validation loss improves, so a late-training divergence (e.g. a NaN loss spike) can never
    destroy a converged model — the caller restores these weights before saving the final
    approximator. Evaluation callers pass no ``run_dir`` and are unaffected.
    """
    import bayesflow as bf
    from omegaconf import OmegaConf

    adapter = build_adapter(cfg.adapter)
    summary_network = build_summary_network(cfg.model.summary_network)
    inference_network = build_inference_network(cfg.model.inference_network)

    standardize = list(OmegaConf.to_container(cfg.training.standardize, resolve=True))

    kwargs = dict(
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network,
        standardize=standardize,
    )
    if run_dir is not None and bool(cfg.training.save_best_weights):
        kwargs.update(
            checkpoint_filepath=run_dir,
            checkpoint_name=BEST_WEIGHTS_NAME,
            save_best_only=True,
            save_weights_only=True,
        )
    return bf.BasicWorkflow(**kwargs)
