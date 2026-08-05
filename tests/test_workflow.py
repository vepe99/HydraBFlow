"""Adapter / network / workflow construction. Skipped if bayesflow isn't installed."""

from __future__ import annotations

import pytest

bf = pytest.importorskip("bayesflow")


def test_build_adapter(cfg):
    from hydrabflow.pipeline.adapter import build_adapter

    adapter = build_adapter(cfg.adapter)
    assert adapter is not None


def test_build_adapter_unconfigured_raises(compose):
    from hydrabflow.pipeline.adapter import build_adapter

    # No registered simulator to derive from and no explicit config: actionable error.
    cfg = compose(["simulator.name=does_not_exist"])
    with pytest.raises(ValueError, match="adapter.inference_variables"):
        build_adapter(cfg.adapter)


def test_build_networks(cfg):
    from hydrabflow.registry import build_inference_network, build_summary_network

    assert build_summary_network(cfg.model.summary_network) is not None
    assert build_inference_network(cfg.model.inference_network) is not None


def test_build_workflow(cfg):
    from hydrabflow.pipeline.workflow import build_workflow

    workflow = build_workflow(cfg)
    assert hasattr(workflow, "approximator")


def test_build_workflow_checkpointing(cfg, tmp_path):
    """Passing run_dir turns on BayesFlow best-weights checkpointing; omitting it leaves it off."""
    from hydrabflow.pipeline.workflow import BEST_WEIGHTS_NAME, build_workflow

    off = build_workflow(cfg)
    assert getattr(off, "checkpoint_filepath", None) is None

    on = build_workflow(cfg, run_dir=str(tmp_path))
    assert on.checkpoint_filepath == str(tmp_path)
    assert on.checkpoint_name == BEST_WEIGHTS_NAME
    assert on.save_best_only is True
    assert on.save_weights_only is True


def test_fusion_when_several_summary_variables(compose):
    """Two observable keys -> one backbone per key behind a FusionNetwork, types from params."""
    from hydrabflow.registry import build_summary_network

    cfg = compose(["simulator=multimodal", "model/summary_network=fusion"])
    assert list(cfg.adapter.summary_variables) == ["x_set", "x_series"]

    net = build_summary_network(cfg.model.summary_network, cfg.adapter.summary_variables)
    assert isinstance(net, bf.networks.FusionNetwork)
    backbones = {k: type(v).__name__ for k, v in net.backbones.items()}
    assert backbones == {"x_set": "SetTransformer", "x_series": "TimeSeriesTransformer"}


def test_fusion_adapter_groups_the_keys(compose):
    from hydrabflow.pipeline.adapter import build_adapter

    cfg = compose(["simulator=multimodal", "model/summary_network=fusion"])
    assert "group" in repr(build_adapter(cfg.adapter)).lower()
