"""Append ``pot_scalars`` (n, 3) = [K_z, mu_l, M200/1e12] to a stream dataset, in place.

Three potential-derived constraints for the rotation-curve branch, each a deterministic function of
the row's stored globals (the host potential is rebuilt as in ``recompute_vcirc_grid.py``):

* ``K_z``  [km^2 s^-2 pc^-1]: vertical acceleration at (R0_Sun, 0, z=1.1 kpc); obs 2.00 +- 0.16.
* ``mu_l`` [mas/yr]: Sgr A* proper motion along l, -(v_c(R0) + V_Sun) / R0; obs -6.379 +- 0.026.
* ``M200`` [1e12 Msun]: TOTAL mass (all components) inside r200 = radius of mean density
  200 rho_crit (H0 from the snapshot, 70.4); obs 1.17 +- 0.21.

The key is written as a new member of the existing (uncompressed) npz with zipfile mode "a", so the
14 GB file is not rewritten. Refuses if ``pot_scalars`` already exists. Shape (n, 3) is ndim 2, so
``flatten_members`` repeats it per member like any group-level array (grouped test sets work too).

Usage (CPU only):
    JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0 python scripts/add_pot_scalars.py DATA.npz [--n-workers 32]
    ... --check 200   # compute on 200 rows, print vs observed, write nothing
"""

from __future__ import annotations

import argparse
import os
import zipfile

import numpy as np
from omegaconf import OmegaConf

from hydrabflow.simulators.stream_agama import _agama, _host_potential, _vcirc
from hydrabflow.simulators.stream_common import G_KPC_KMS2_MSUN

KEY = "pot_scalars"
OBS = np.array([2.00, -6.379, 1.17])
ERR = np.array([0.16, 0.026, 0.21])
KMS_KPC_TO_MASYR = 1.0 / 4.740470463533348   # (km/s)/kpc -> mas/yr
Z_KZ_KPC = 1.1


def _m200_total(pot, h0_kms_mpc: float) -> float:
    h0 = h0_kms_mpc / 1e3  # km/s/kpc
    target = (4.0 / 3.0) * np.pi * 200.0 * 3.0 * h0 * h0 / (8.0 * np.pi * G_KPC_KMS2_MSUN)

    def f(r):
        return float(pot.enclosedMass(r)) - target * r**3

    lo, hi = 10.0, 1000.0
    if f(lo) < 0 or f(hi) > 0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return float(pot.enclosedMass(0.5 * (lo + hi)))


def _worker(rows: list[dict], pot_cfg, h0: float) -> np.ndarray:
    agama = _agama()
    agama.setNumThreads(1)
    out = np.full((len(rows), 3), np.nan)
    for i, p in enumerate(rows):
        try:
            pot = _host_potential(agama, p, pot_cfg)
            r0 = p["R0_Sun"]
            kz = -pot.force(np.array([r0, 0.0, Z_KZ_KPC]))[2] / 1e3
            vc = _vcirc(pot, np.array([r0]))[0]
            mu_l = -(vc + p["V_Sun"]) / r0 * KMS_KPC_TO_MASYR
            out[i] = kz, mu_l, _m200_total(pot, h0) / 1e12
        except Exception:
            pass  # NaN row; drop_nan removes it
    return out


def compute(scalars: dict, idx: np.ndarray, pot_cfg, h0: float, n_workers: int) -> np.ndarray:
    from joblib import Parallel, delayed

    rows = [{k: float(v[i]) for k, v in scalars.items()} for i in idx]
    n_jobs = max(1, min(n_workers, len(rows)))
    res = Parallel(n_jobs=n_jobs)(delayed(_worker)(rows[i::n_jobs], pot_cfg, h0) for i in range(n_jobs))
    out = np.empty((len(rows), 3))
    for i, r in enumerate(res):
        out[i::n_jobs] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--n-workers", type=int, default=32)
    ap.add_argument("--check", type=int, default=0, help="compute N rows, print, write nothing")
    args = ap.parse_args()

    with zipfile.ZipFile(args.dataset) as z:
        if f"{KEY}.npy" in z.namelist() and not args.check:
            print(f"{args.dataset}: {KEY} already present, nothing to do")
            return

    from hydrabflow.registry import get_simulator

    snap = OmegaConf.load(os.path.join(os.path.splitext(args.dataset)[0] + ".hydra", "config.yaml"))
    pot_cfg = get_simulator(snap.simulator)._pot_cfg
    h0 = float((pot_cfg or {}).get("halo_H0_kms_mpc", 70.4))

    # only the (n, 1) scalar columns: never load sim_data_projected
    raw = np.load(args.dataset)
    n = raw["R0_Sun"].shape[0]
    scalars = {}
    for k in raw.files:
        if k == "sim_data_projected":
            continue
        v = raw[k]
        if v.shape == (n, 1):
            scalars[k] = v.reshape(n)

    idx = (np.random.default_rng(0).choice(n, size=min(args.check, n), replace=False)
           if args.check else np.arange(n))
    out = compute(scalars, idx, pot_cfg, h0, args.n_workers)

    # cross-check mu_l against the simulator's own v_c(R0)
    if "vc_R0_derived" in scalars:
        mu_ref = -(scalars["vc_R0_derived"][idx] + scalars["V_Sun"][idx]) / scalars["R0_Sun"][idx] * KMS_KPC_TO_MASYR
        print(f"mu_l vs vc_R0_derived: max |diff| {np.nanmax(np.abs(out[:, 1] - mu_ref)):.2e} mas/yr")
    for j, name in enumerate(["K_z [km2/s2/pc]", "mu_l [mas/yr]", "M200 [1e12 Msun]"]):
        q = np.nanpercentile(out[:, j], [5, 50, 95])
        frac = np.nanmean(np.abs(out[:, j] - OBS[j]) < ERR[j])
        print(f"{name:18s} p5/p50/p95 {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f}   obs {OBS[j]} +- {ERR[j]}"
              f"   within 1 sigma {frac:.1%}")
    print(f"NaN rows: {int(np.isnan(out).any(axis=1).sum())} of {len(idx)}")
    if args.check:
        return

    with zipfile.ZipFile(args.dataset, mode="a", allowZip64=True) as z:
        with z.open(f"{KEY}.npy", "w", force_zip64=True) as f:
            np.lib.format.write_array(f, out.astype(np.float32))
    print(f"appended {KEY} {out.shape} to {args.dataset}")


if __name__ == "__main__":
    main()
