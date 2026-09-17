"""Assemble the BayesFlow workflow (adapter + summary network + inference network) from config.

``composition.level=none`` (the default) gives a ``bf.BasicWorkflow``. ``global`` / ``local`` give a
``bf.CompositionalWorkflow``: training is identical (rows are independent), but it exposes
``compositional_sample`` / ``ancestral_sample`` so evaluation can pool the exchangeable members of
a group (e.g. several streams constraining one galactic potential).
"""

from __future__ import annotations

from typing import Any

from hydrabflow.pipeline.adapter import build_adapter, composition_level
from hydrabflow.registry import build_inference_network, build_summary_network

#: Basename of the best-weights checkpoint BayesFlow writes during training.
BEST_WEIGHTS_NAME = "approximator_best"


def build_workflow(cfg, run_dir: str | None = None) -> Any:
    """Build the workflow from the root ``cfg``.

    ``run_dir`` (passed by train and tune) turns on BayesFlow's best-weights checkpointing, so a
    late divergence cannot destroy a converged model; the caller restores those weights before
    saving. Evaluation callers pass nothing and are unaffected.
    """
    import bayesflow as bf
    from omegaconf import OmegaConf

    inference_network = build_inference_network(cfg.model.inference_network, cfg.model)
    # CompositionalWorkflow accepts only a DiffusionModel. Any other net at composition=global
    # (e.g. coupling_flow, used to train standalone summary nets) trains identically under
    # BasicWorkflow; the adapter derivation still selects the level's parameters, and evaluate
    # skips the compositional stage when `compositional_sample` is absent.
    if composition_level(cfg) == "none" or not isinstance(
        inference_network, bf.networks.DiffusionModel
    ):
        workflow_cls = bf.BasicWorkflow
    else:
        from hydrabflow.pipeline._bf_patches import apply_bayesflow_patches

        apply_bayesflow_patches()
        workflow_cls = bf.CompositionalWorkflow

    kwargs = dict(
        adapter=build_adapter(cfg.adapter),
        summary_network=build_summary_network(
            cfg.model.summary_network, cfg.adapter.summary_variables
        ),
        inference_network=inference_network,
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
    return workflow_cls(**kwargs)
