"""Stage 1: dataset generation.

Samples the prior and runs the forward model in chunks, writing one aggregated ``.npz`` to
``data.data_dir/data.dataset_name``. Generalizes the reference ``simulate_main.py`` (without the
multistream / global-local specialization). Each row of the dataset is one (parameters,
observation) pair: the union of ``sample_prior`` and ``simulate`` outputs.
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


def run_simulation(cfg) -> str:
    """Generate the dataset described by ``cfg`` and return its path."""
    seed_everything(cfg.seed)
    simulator = get_simulator(cfg.simulator)
    log.info(
        "Simulator '%s': params=%s observables=%s",
        cfg.simulator.name,
        simulator.parameter_names,
        simulator.observable_keys,
    )

    # Checkpoint each chunk to disk as it completes (resumable): a crash only costs the in-flight
    # chunk, and re-running skips the chunks already on disk. See io.run_chunked.
    out_path = os.path.join(cfg.data.data_dir, cfg.data.dataset_name)
    io.run_chunked(
        out_path,
        n_total=int(cfg.data.n_simulations),
        chunk=int(cfg.data.chunk_size),
        sample_fn=simulator.sample,
        base_seed=int(cfg.seed),
        desc="simulating",
    )

    # Traceability: copy Hydra's `.hydra/` config snapshot next to the dataset, keyed by the
    # dataset filename so training and test sets in the same data_dir don't clobber each other.
    stem = os.path.splitext(cfg.data.dataset_name)[0]
    snapshot = save_config_snapshot(cfg.data.data_dir, stem)
    if snapshot:
        log.info("Saved config snapshot -> %s", snapshot)
    return out_path


cli = make_cli(run_simulation)


if __name__ == "__main__":
    cli()
