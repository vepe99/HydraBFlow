"""Copy a stream dataset with its ``vcirc_kms`` recomputed on a simulator config's rotation-curve grid.

The model rotation curve is a deterministic function of the stored global-potential columns, so
any dataset (with or without an existing ``vcirc_kms``) can be re-gridded by rebuilding each row's
host potential (~18 ms/row) -- no stream re-simulation. The potential model (m200_c halo, gas disks,
exponential disks, ...) is the one the dataset was simulated with: it is read from the dataset's own
``<stem>.hydra/`` snapshot; without a snapshot the legacy potential (``_host_potential`` default) is
used. Every scalar ``(n, 1)`` column is passed as the row, so free and identity parameters are both
seen. Works for flat and grouped (``simulate_multistream``) files alike: the globals are ``(n, 1)``
and ``vcirc_kms`` ``(n, R, 1)`` in both.

``--check N`` first recomputes N rows on the dataset's OLD grid and compares with the stored
``vcirc_kms``; it aborts if they disagree (i.e. if the potential was not rebuilt faithfully).

Usage (CPU only -- importing hydrabflow.simulators pulls in JAX):
    JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0 python scripts/recompute_vcirc_grid.py IN.npz \
        --sim-yaml conf/simulator/<new grid>.yaml --out OUT.npz [--n-workers 32] [--check 2000]
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from omegaconf import OmegaConf

from hydrabflow.simulators.stream_agama import _agama, _host_potential, _vcirc


def _worker(rows: list[dict], obs_r: np.ndarray, pot_cfg: dict | None) -> np.ndarray:
    agama = _agama()
    agama.setNumThreads(1)
    out = np.full((len(rows), len(obs_r)), np.nan)
    for i, p in enumerate(rows):
        try:
            out[i] = _vcirc(_host_potential(agama, p, pot_cfg), obs_r)
        except Exception:
            pass  # leave NaN; drop_nan removes the row downstream
    return out


def recompute(data: dict, idx: np.ndarray, obs_r: np.ndarray, pot_cfg: dict | None,
              n_workers: int) -> np.ndarray:
    from joblib import Parallel, delayed

    n = len(data["j"])
    scalars = {k: v.reshape(n) for k, v in data.items() if v.shape == (n, 1)}
    rows = [{k: float(v[i]) for k, v in scalars.items()} for i in idx]
    n_jobs = max(1, min(n_workers, len(rows)))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(rows[i::n_jobs], obs_r, pot_cfg) for i in range(n_jobs))
    vc = np.empty((len(rows), len(obs_r)))
    for i, r in enumerate(res):
        vc[i::n_jobs] = r
    return vc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--sim-yaml", required=True, help="simulator yaml with obs_r_kpc/obs_vc_kms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-workers", type=int, default=32)
    ap.add_argument("--check", type=int, default=2000, help="rows to verify on the old grid (0 = skip)")
    args = ap.parse_args()

    new = OmegaConf.load(args.sim_yaml).params
    r = np.asarray(new.obs_r_kpc, dtype=float)
    vc_obs = np.asarray(new.obs_vc_kms, dtype=float)

    src_hydra = os.path.splitext(os.path.abspath(args.dataset))[0] + ".hydra"
    pot_cfg, old_r = None, None
    if os.path.isdir(src_hydra):
        from hydrabflow.registry import get_simulator

        snap = OmegaConf.load(os.path.join(src_hydra, "config.yaml"))
        sim = get_simulator(snap.simulator)
        pot_cfg, old_r = sim._pot_cfg, np.asarray(sim.obs_r_kpc, dtype=float)
    print(f"potential config: {pot_cfg or 'legacy (no .hydra snapshot)'}")

    raw = np.load(args.dataset, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    n = len(data["j"])
    old = data.get("vcirc_kms")

    if args.check and old is not None and old_r is not None and old.shape[1] == len(old_r):
        idx = np.random.default_rng(0).choice(n, size=min(args.check, n), replace=False)
        vc = recompute(data, idx, old_r, pot_cfg, args.n_workers)
        ref = old[idx, :, 0]
        ok = np.isfinite(ref) & np.isfinite(vc)
        dev = np.abs(vc[ok] - ref[ok]) / ref[ok]
        print(f"check on old grid ({len(idx)} rows): max frac dev {dev.max():.2e}, "
              f"NaN mismatch {int((np.isfinite(ref) != np.isfinite(vc)).sum())}")
        if dev.max() > 1e-4:
            raise SystemExit("recomputed curve does not reproduce the stored vcirc_kms -- aborting")

    print(f"{args.dataset}: {n} rows; vcirc_kms {None if old is None else old.shape} -> ({n}, {len(r)}, 1)")
    vc = recompute(data, np.arange(n), r, pot_cfg, args.n_workers)
    data["vcirc_kms"] = vc[..., None].astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **data)
    if os.path.isdir(src_hydra):
        dst = os.path.splitext(args.out)[0] + ".hydra"
        shutil.copytree(src_hydra, dst, dirs_exist_ok=True)
        with open(os.path.join(dst, "vcirc_regrid.txt"), "w") as f:
            f.write(f"vcirc_kms recomputed on the grid of {args.sim_yaml} ({len(r)} radii) "
                    f"from {os.path.abspath(args.dataset)}\n")

    n_nan = int(np.isnan(vc).any(axis=1).sum())
    med = np.nanmedian(vc, axis=0)
    print(f"Wrote {args.out}: {n_nan} NaN curves; dataset-median curve vs observed: "
          f"median |frac dev| = {np.median(np.abs(med - vc_obs) / vc_obs):.3f}")


if __name__ == "__main__":
    main()
