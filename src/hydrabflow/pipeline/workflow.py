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


def build_workflow(cfg) -> Any:
    """Build a ``bf.BasicWorkflow`` (or ``bf.CompositionalWorkflow``) from the root ``cfg``."""
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

    return workflow_cls(
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network,
        standardize=standardize,
    )
