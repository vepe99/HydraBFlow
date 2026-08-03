"""Turn a ``run(cfg)`` function into the stage's Hydra console entry point."""

from __future__ import annotations

import os
from typing import Callable

import hydra

from hydrabflow.config import register_configs
from hydrabflow.pipeline.adapter import fill_adapter_from_simulator

CONF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "conf"))


def make_cli(run_fn: Callable) -> Callable[[], None]:
    register_configs()

    def stage(cfg) -> None:
        # The simulator stays the single source of truth for its variable names.
        fill_adapter_from_simulator(cfg)
        run_fn(cfg)

    # Hydra names the job (and its log file) after the task function's module, so borrow the
    # stage's: `train.log`, not `_app.log`.
    stage.__module__ = run_fn.__module__
    return hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")(stage)
