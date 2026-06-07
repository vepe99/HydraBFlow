"""Stage 1: dataset generation.

Samples the prior and runs the forward model in chunks, writing one aggregated ``.npz`` to
``data.data_dir/data.dataset_name``. Generalizes the reference ``simulate_main.py`` (without the
multistream / global-local specialization). Each row of the dataset is one (parameters,
observation) pair: the union of ``sample_prior`` and ``simulate`` outputs.
"""

from __future__ import annotations

import os

from tqdm import tqdm

from hydraflow.pipeline import io
from hydraflow.pipeline._app import make_cli
from hydraflow.simulators.registry import get_simulator
from hydraflow.utils.logging import get_logger
from hydraflow.utils.paths import save_config_snapshot
from hydraflow.utils.seed import seed_everything

log = get_logger(__name__)


def run_simulation(cfg) -> str:
    """Generate the dataset described by ``cfg`` and return its path."""
    rng = seed_everything(cfg.seed)
    simulator = get_simulator(cfg.simulator)
    log.info(
        "Simulator '%s': params=%s observables=%s",
        cfg.simulator.name,
        simulator.parameter_names,
        simulator.observable_keys,
    )

    n_total = int(cfg.data.n_simulations)
    chunk = int(cfg.data.chunk_size)
    chunks = []
    for start in tqdm(range(0, n_total, chunk), desc="simulating"):
        n = min(chunk, n_total - start)
        chunks.append(simulator.sample(n, rng))

    data = io.concatenate_chunks(chunks)
    out_path = os.path.join(cfg.data.data_dir, cfg.data.dataset_name)
    io.save_dataset(out_path, data)

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
