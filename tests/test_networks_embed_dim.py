"""Attention width via ``params.embed_dim_multiplier`` (width per head)."""

from __future__ import annotations

import pytest


def test_embed_dim_multiplier_takes_precedence():
    """The multiplier yields num_heads * multiplier; without it the raw embed_dim is used."""
    from omegaconf import OmegaConf

    from hydrabflow.networks.factory import _embed_dim

    raw = OmegaConf.create({"num_heads": 5, "embed_dim": 128, "params": {}})
    assert _embed_dim(raw) == 128

    scaled = OmegaConf.create(
        {"num_heads": 5, "embed_dim": 128, "params": {"embed_dim_multiplier": 16}}
    )
    assert _embed_dim(scaled) == 80  # 5 * 16, divisible by num_heads


@pytest.mark.parametrize("num_heads", [3, 5, 7])
def test_transformer_backbones_build_for_any_head_count(compose, num_heads):
    """A head count that does not divide the default embed_dim used to fail; with the multiplier
    every draw is valid — this is what makes the tuning search space usable."""
    from hydrabflow.networks.factory import build_summary_network

    for net_type in ("set_transformer", "time_series_transformer"):
        cfg = compose(
            [
                f"model/summary_network={net_type}",
                f"model.summary_network.num_heads={num_heads}",
                "+model.summary_network.params.embed_dim_multiplier=16",
            ]
        )
        assert build_summary_network(cfg.model.summary_network) is not None
