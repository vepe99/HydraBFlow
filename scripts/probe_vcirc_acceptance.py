"""Measure the vcirc-rejection acceptance rate of a stream simulator's prior (calibration helper).

Draws a batch of raw prior samples (no stream integration) and reports, per rejection band and
combined, the fraction of global-potential draws that pass the ``vcirc_rejection`` cut. Use it to
calibrate the cut (e.g. the Huang ``max_n_sigma``) and to size a dataset run before committing to
it: at a combined acceptance ``a``, generating ``N`` accepted rows screens about ``N/a`` potentials
(~18 ms each), and the simulator aborts if it must draw > 5M without filling ``N``.

Usage:
    python scripts/probe_vcirc_acceptance.py [--simulator stream_agama_spray_huang]
        [--n-draws 20000] [--n-workers 24] [--seed 0]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import hydrabflow  # noqa: F401  (pins the JAX backend + triggers simulator discovery)
from hydrabflow.simulators.registry import get_simulator
from hydrabflow.simulators.stream_agama import _agama, _host_potential, _vcirc
from hydrabflow.simulators.stream_common import sample_stream_prior


def _band_pass_worker(rows: list, obs_r: np.ndarray, bands: list) -> np.ndarray:
    """(n_rows, n_bands) bool: does each row's model curve pass each band? (one vc eval per row)."""
    agama = _agama()
    agama.setNumThreads(1)
    out = np.zeros((len(rows), len(bands)), dtype=bool)
    for i, p in enumerate(rows):
        try:
            vc = _vcirc(_host_potential(agama, p), obs_r)
        except Exception:
            continue
        for bi, b in enumerate(bands):
            v = vc[b["bin_mask"]]
            if np.isnan(v).any():
                continue
            denom = b["sigma"] if b["criterion"] == "sigma" else b["vc_ref"]
            dev = np.abs(v - b["vc_ref"]) / denom
            reduce = np.median if b["stat"] == "median" else np.max
            out[i, bi] = bool(reduce(dev) < b["threshold"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe vcirc-rejection prior acceptance.")
    ap.add_argument("--simulator", default="stream_agama_spray_huang")
    ap.add_argument("--n-draws", type=int, default=20000)
    ap.add_argument("--n-workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir

    from hydrabflow.config.schema import register_configs

    register_configs()  # register the structured-config schemas the defaults list references
    conf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config", overrides=[f"simulator={args.simulator}"])
    sim = get_simulator(cfg.simulator)

    cfg_rej = sim._vcirc_rejection
    if cfg_rej is None:
        raise SystemExit(f"simulator '{args.simulator}' has no vcirc_rejection to probe")
    obs_r = sim.obs_r_kpc
    bands = sim._build_accept_bands(cfg_rej, obs_r)
    band_names = [b.get("name", f"band{i}") for i, b in enumerate(cfg_rej.get("bands", [{}]))]

    rng = np.random.default_rng(args.seed)
    draws = sample_stream_prior(sim._priors_global, sim._priors_local, sim.target_streams,
                                args.n_draws, rng)
    gkeys = list(sim._priors_global)
    rows = [{k: float(np.asarray(draws[k]).reshape(args.n_draws, -1)[i, 0]) for k in gkeys}
            for i in range(args.n_draws)]

    from joblib import Parallel, delayed
    n_jobs = min(args.n_workers, len(rows))
    chunks = [rows[i::n_jobs] for i in range(n_jobs)]
    res = Parallel(n_jobs=n_jobs)(delayed(_band_pass_worker)(c, obs_r, bands) for c in chunks)
    passed = np.zeros((args.n_draws, len(bands)), dtype=bool)
    for i, r in enumerate(res):
        passed[i::n_jobs] = r

    combined = passed.all(axis=1)
    a = combined.mean()
    print(f"simulator={args.simulator}  n_draws={args.n_draws}  grid={len(obs_r)} radii")
    for bi, name in enumerate(band_names):
        b = bands[bi]
        crit = f"{b['stat']} {b['criterion']} < {b['threshold']}"
        print(f"  band '{name}':  pass {passed[:, bi].mean():6.2%}   ({crit})")
    print(f"  COMBINED (all bands): {a:6.2%}")
    if a > 0:
        print(f"  -> ~{1/a:.1f} potential screens per accepted row; "
              f"30000 rows ~= {30000/a/1e3:.0f}k screens "
              f"(abort guard at 5000k draws => min safe acceptance {30000/5e6:.2%}).")


if __name__ == "__main__":
    main()
