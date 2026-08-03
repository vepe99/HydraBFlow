"""Attention width is configured per head, so it is always divisible by the head count."""

from __future__ import annotations

import pytest


def test_embed_dim_is_per_head():
    from omegaconf import OmegaConf

    from hydrabflow.networks.factory import _embed_dim

    assert _embed_dim(OmegaConf.create({"num_heads": 5, "embed_dim_per_head": 16})) == 80


@pytest.mark.parametrize("num_heads", [3, 5, 7])
def test_transformer_backbones_build_for_any_head_count(compose, num_heads):
    """A head count that does not divide a raw embed_dim used to fail; per-head width makes every
    draw valid — this is what makes the tuning search space usable."""
    pytest.importorskip("bayesflow")
    from hydrabflow.networks.factory import build_summary_network

    for net_type in ("set_transformer", "time_series_transformer"):
        cfg = compose(
            [
                f"model/summary_network={net_type}",
                f"model.summary_network.num_heads={num_heads}",
            ]
        )
        assert build_summary_network(cfg.model.summary_network) is not None
