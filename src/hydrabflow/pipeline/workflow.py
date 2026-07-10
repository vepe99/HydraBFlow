"""Assemble the BayesFlow workflow from config.

Single-level inference (the default, ``composition.level=none``) uses ``bf.BasicWorkflow``.
Compositional score modeling (``level=global`` / ``level=local``) uses
``bf.CompositionalWorkflow``: training is identical (rows are treated independently), but the
workflow exposes ``compositional_sample`` / ``ancestral_sample`` so evaluation can pool the
exchangeable group members (e.g. several streams constraining one galactic potential).
"""

from __future__ import annotations

from typing import Any

from hydrabflow.networks.factory import build_inference_network, build_summary_network
from hydrabflow.pipeline.adapter import build_adapter


#: Basename (sans extension) of the best-weights checkpoint BayesFlow writes during training.
BEST_WEIGHTS_NAME = "approximator_best"


def build_workflow(cfg, run_dir: str | None = None) -> Any:
    """Build a ``bf.BasicWorkflow`` (or ``bf.CompositionalWorkflow``) from the root ``cfg``.

    ``run_dir`` (set only by the train stage) turns on BayesFlow's built-in best-weights
    checkpointing: it writes ``<run_dir>/approximator_best.weights.h5`` whenever the monitored
    validation loss improves, so a late-training divergence (e.g. a NaN loss spike) can never
    destroy a converged model — the train stage restores these weights before saving the final
    approximator. Evaluation/tuning callers pass no ``run_dir`` and are unaffected.
    """
    import bayesflow as bf
    from omegaconf import OmegaConf

    adapter = build_adapter(cfg.adapter)
    summary_network = build_summary_network(cfg.model.summary_network)
    inference_network = build_inference_network(cfg.model.inference_network)

    standardize = list(OmegaConf.to_container(cfg.training.standardize, resolve=True))

    level = str(getattr(getattr(cfg, "composition", None), "level", "none") or "none")
    if level == "none":
        workflow_cls = bf.BasicWorkflow
    else:
        from hydrabflow.pipeline._bf_patches import apply_bayesflow_patches

        apply_bayesflow_patches()
        workflow_cls = bf.CompositionalWorkflow

    kwargs = dict(
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network,
        standardize=standardize,
    )
    if run_dir is not None and bool(getattr(cfg.training, "save_best_weights", True)):
        kwargs.update(
            checkpoint_filepath=run_dir,
            checkpoint_name=BEST_WEIGHTS_NAME,
            save_best_only=True,
            save_weights_only=True,
        )
    return workflow_cls(**kwargs)
