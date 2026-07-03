"""Registry resolution + skeleton-simulator behavior."""

from __future__ import annotations

import numpy as np
import pytest


def test_simulator_registry_has_shipped_simulators():
    from hydrabflow.simulators.registry import available_simulators

    assert "skeleton" in available_simulators()
    assert "two_moons" in available_simulators()


def test_skeleton_simulator_raises(compose):
    from hydrabflow.simulators.registry import get_simulator

    cfg = compose(["simulator=skeleton"], fill=False)
    sim = get_simulator(cfg.simulator)
    assert sim.parameter_names == ["theta1", "theta2"]
    assert sim.observable_keys == ["x"]
    with pytest.raises(NotImplementedError):
        sim.sample_prior(4, np.random.default_rng(0))


def test_preprocess_registry():
    from hydrabflow.preprocessing.registry import available_steps

    for name in ("drop_nan", "train_val_split", "standardize", "cast_dtype", "select_keys"):
        assert name in available_steps()


def test_augmentation_registry_builds(cfg):
    from hydrabflow.augmentation.registry import build_augmentations

    rng = np.random.default_rng(0)
    assert build_augmentations(cfg.augmentation, rng) == []  # empty by default


def test_unknown_simulator_errors(cfg):
    from hydrabflow.simulators.registry import get_simulator

    cfg.simulator.name = "does_not_exist"
    with pytest.raises(KeyError):
        get_simulator(cfg.simulator)


def test_network_registries_list_available_on_unknown_type(cfg):
    from hydrabflow.networks.factory import build_inference_network, build_summary_network

    cfg.model.summary_network.type = "does_not_exist"
    with pytest.raises(ValueError, match="set_transformer"):
        build_summary_network(cfg.model.summary_network)

    cfg.model.inference_network.type = "does_not_exist"
    with pytest.raises(ValueError, match="flow_matching"):
        build_inference_network(cfg.model.inference_network)


def test_custom_network_builder_registers(cfg):
    from hydrabflow.networks.factory import build_summary_network, register_summary_network

    sentinel = object()

    @register_summary_network("_test_custom")
    def _custom(net_cfg):
        return sentinel

    try:
        cfg.model.summary_network.type = "_test_custom"
        assert build_summary_network(cfg.model.summary_network) is sentinel
    finally:
        from hydrabflow.networks import factory

        factory._SUMMARY_BUILDERS.pop("_test_custom", None)
