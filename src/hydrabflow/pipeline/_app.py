"""Shared Hydra-app boilerplate for the five run stages."""

from __future__ import annotations

import os
from typing import Callable


def conf_path() -> str:
    """Absolute path to the repo-root ``conf/`` directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", "conf"))


def make_cli(run_fn: Callable) -> Callable[[], None]:
    """Wrap a ``run_fn(cfg)`` into a Hydra console entry point.

    Registers the structured configs, then dispatches the root ``config`` to ``run_fn``.
    """

    def cli() -> None:
        import hydra

        from hydrabflow.config import register_configs
        from hydrabflow.pipeline.adapter import fill_adapter_from_simulator

        register_configs()

        @hydra.main(version_base=None, config_path=conf_path(), config_name="config")
        def _main(cfg):
            # Empty adapter variable lists are derived from the simulator's declaration, so the
            # simulator class stays the single source of truth for its names/keys.
            fill_adapter_from_simulator(cfg)
            run_fn(cfg)

        _main()

    return cli
