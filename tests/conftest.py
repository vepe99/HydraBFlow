"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest

CONF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))


def compose_cfg(overrides=None, fill=True):
    """Compose the root config with the structured schemas registered.

    ``fill=True`` mirrors the CLI entry points: empty adapter variable lists are derived from
    the selected simulator (see ``pipeline.adapter.fill_adapter_from_simulator``).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from hydrabflow.config import register_configs
    from hydrabflow.pipeline.adapter import fill_adapter_from_simulator

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    if fill:
        fill_adapter_from_simulator(cfg)
    return cfg


@pytest.fixture
def cfg():
    return compose_cfg()


@pytest.fixture
def compose():
    """Expose the composer so tests can build configs with custom overrides."""
    return compose_cfg
