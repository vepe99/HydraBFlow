"""Stage 1: dataset generation.

Samples the prior and runs the forward model in chunks into one ``.npz`` at
``data.data_dir/data.dataset_name``. Each row is one (parameters, observation) pair.

The one stage that does *not* write to the Hydra run dir: a dataset is a shared input reused by many
runs, so its path is configured explicitly and ``save_config_snapshot`` copies the resolved config
next to it by hand. Grouped (compositional) datasets are written by ``simulate_multistream``.
"""

from __future__ import annotations

import logging
import os

from hydrabflow.pipeline import io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.registry import get_simulator
from hydrabflow.utils.paths import save_config_snapshot
from hydrabflow.utils.seed import seed_everything

log = logging.getLogger(__name__)


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

    out_path = os.path.join(cfg.data.data_dir, cfg.data.dataset_name)
    io.run_chunked(
        out_path,
        n_total=int(cfg.data.n_simulations),
        chunk=int(cfg.data.chunk_size),
        sample_fn=simulator.sample,
        base_seed=int(cfg.seed),
    )

    # Traceability: copy Hydra's `.hydra/` config snapshot next to the dataset, keyed by the
    # dataset filename so training and test sets in the same data_dir don't clobber each other.
    snapshot = save_config_snapshot(
        cfg.data.data_dir, os.path.splitext(cfg.data.dataset_name)[0]
    )
    if snapshot:
        log.info("Saved config snapshot -> %s", snapshot)
    return out_path


cli = make_cli(run_simulation)


if __name__ == "__main__":
    cli()
