"""Registry resolution: unknown names fail loudly, custom builders plug in."""

from __future__ import annotations

import numpy as np
import pytest


def test_shipped_components_are_registered():
    """Every extension point is a Registry filled on discovery (registry.py)."""
    from hydrabflow.registry import AUGMENTATIONS, SIMULATORS, STEPS

    for registry in (SIMULATORS, STEPS, AUGMENTATIONS):
        registry.discover()  # normally triggered lazily by the first .get()

    assert {"two_moons", "multimodal", "stream_agama", "stream_gala"} <= set(SIMULATORS.items)
    assert {"drop_nan", "train_val_split", "standardize"} <= set(STEPS.items)
    assert "gaussian_noise" in AUGMENTATIONS.items


def test_augmentation_registry_builds(cfg):
    from hydrabflow.registry import build_augmentations

    assert build_augmentations(cfg.augmentation, np.random.default_rng(0)) == []  # empty default


def test_unknown_simulator_errors(cfg):
    from hydrabflow.registry import get_simulator

    cfg.simulator.name = "does_not_exist"
    with pytest.raises(KeyError):
        get_simulator(cfg.simulator)


def test_unknown_preprocess_step_errors(cfg):
    from hydrabflow.registry import build_pipeline

    cfg.preprocessing.steps = [{"name": "does_not_exist"}]
    with pytest.raises(KeyError, match="drop_nan"):  # the message lists what *is* registered
        build_pipeline(cfg.preprocessing)


def test_network_registries_list_available_on_unknown_type(cfg):
    from hydrabflow.registry import build_inference_network, build_summary_network

    cfg.model.summary_network.type = "does_not_exist"
    with pytest.raises(KeyError, match="set_transformer"):
        build_summary_network(cfg.model.summary_network)

    cfg.model.inference_network.type = "does_not_exist"
    with pytest.raises(KeyError, match="flow_matching"):
        build_inference_network(cfg.model.inference_network)


def test_custom_network_builder_registers(cfg):
    """Adding an experimental architecture = one decorated function, no infrastructure edits."""
    from hydrabflow.registry import build_summary_network, register_summary_network

    sentinel = object()

    @register_summary_network("_test_custom")
    def _custom(net_cfg):
        return sentinel

    try:
        cfg.model.summary_network.type = "_test_custom"
        assert build_summary_network(cfg.model.summary_network) is sentinel
    finally:
        from hydrabflow.registry import SUMMARY_NETWORKS

        SUMMARY_NETWORKS.items.pop("_test_custom", None)
