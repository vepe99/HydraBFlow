"""Turn a ``run(cfg)`` function into the stage's Hydra console entry point."""

from __future__ import annotations

import os
import sys
from typing import Callable

import hydra

from hydrabflow.config import register_configs
from hydrabflow.pipeline.adapter import fill_adapter_from_simulator, fill_stream_grid_from_simulator

CONF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "conf"))


def make_cli(run_fn: Callable) -> Callable[[], None]:
    register_configs()

    def stage(cfg) -> None:
        # The simulator stays the single source of truth for its variable names (and, for the
        # stream simulators, for the rotation-curve grid the training-time components must match).
        fill_adapter_from_simulator(cfg)
        fill_stream_grid_from_simulator(cfg)
        run_fn(cfg)

    entry = hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")(stage)
    module = run_fn.__module__
    if module == "__main__":  # `python -m hydrabflow.pipeline.<stage>`
        spec = getattr(sys.modules["__main__"], "__spec__", None)
        module = spec.name if spec else os.path.splitext(os.path.basename(sys.argv[0]))[0]
    job_name = module.rsplit(".", 1)[-1]

    def cli() -> None:
        # Hydra names the job (and its log file) after the file that called hydra.main -- this
        # one -- so pass the stage's name explicitly: `train.log`, not `_app.log`.
        if not any(a.startswith("hydra.job.name=") for a in sys.argv[1:]):
            sys.argv.append(f"hydra.job.name={job_name}")
        entry()

    return cli
