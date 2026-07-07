"""Stage 1b: compositional (grouped) dataset generation.

Like :mod:`simulate`, but each row is a *group* of observations sharing one global-parameter
draw (e.g. all target streams evolved in the same Milky Way potential), produced by
``BaseSimulator.sample_compositional``. These datasets are what the compositional evaluation
stage consumes (global posterior via compositional sampling over the group members).
Generalizes the reference ``simulate_multistream_main.py``.

Shape convention (``m`` = group members): globals ``(n, 1)``; locals / context keys
``(n, m, 1)``; member observables ``(n, m, *event_shape)``.
"""

from __future__ import annotations

import os

from hydrabflow.pipeline import io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.simulators.registry import get_simulator
from hydrabflow.utils.logging import get_logger
from hydrabflow.utils.paths import save_config_snapshot
from hydrabflow.utils.seed import seed_everything

log = get_logger(__name__)


def run_multistream_simulation(cfg) -> str:
    """Generate the compositional dataset described by ``cfg`` and return its path."""
    seed_everything(cfg.seed)
    simulator = get_simulator(cfg.simulator)
    log.info(
        "Simulator '%s' (compositional): global=%s local=%s context=%s observables=%s",
        cfg.simulator.name,
        simulator.global_parameter_names,
        simulator.local_parameter_names,
        simulator.context_keys,
        simulator.observable_keys,
    )

    # Checkpoint each chunk to disk as it completes (resumable): a crash only costs the in-flight
    # chunk, and re-running skips the chunks already on disk. See io.run_chunked.
    out_path = os.path.join(cfg.data.data_dir, cfg.data.dataset_name)
    io.run_chunked(
        out_path,
        n_total=int(cfg.data.n_simulations),
        chunk=int(cfg.data.chunk_size),
        sample_fn=simulator.sample_compositional,
        base_seed=int(cfg.seed),
        desc="simulating (compositional)",
    )

    stem = os.path.splitext(cfg.data.dataset_name)[0]
    snapshot = save_config_snapshot(cfg.data.data_dir, stem)
    if snapshot:
        log.info("Saved config snapshot -> %s", snapshot)
    return out_path


cli = make_cli(run_multistream_simulation)


if __name__ == "__main__":
    cli()
