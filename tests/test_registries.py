"""Registry resolution + skeleton-simulator behavior."""

from __future__ import annotations

import numpy as np
import pytest


def test_simulator_registry_has_skeleton():
    from hydraflow.simulators.registry import available_simulators

    assert "skeleton" in available_simulators()


def test_skeleton_simulator_raises(cfg):
    from hydraflow.simulators.registry import get_simulator

    sim = get_simulator(cfg.simulator)
    assert sim.parameter_names == ["theta1", "theta2"]
    assert sim.observable_keys == ["x"]
    with pytest.raises(NotImplementedError):
        sim.sample_prior(4, np.random.default_rng(0))


def test_preprocess_registry():
    from hydraflow.preprocessing.registry import available_steps

    for name in ("drop_nan", "train_val_split", "standardize", "cast_dtype", "select_keys"):
        assert name in available_steps()


def test_augmentation_registry_builds(cfg):
    from hydraflow.augmentation.registry import build_augmentations

    rng = np.random.default_rng(0)
    assert build_augmentations(cfg.augmentation, rng) == []  # empty by default


def test_unknown_simulator_errors(cfg):
    from hydraflow.simulators.registry import get_simulator

    cfg.simulator.name = "does_not_exist"
    with pytest.raises(KeyError):
        get_simulator(cfg.simulator)
