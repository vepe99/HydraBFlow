"""Copy a stream dataset with its ``vcirc_kms`` recomputed on a simulator config's rotation-curve grid.

The model rotation curve is a deterministic function of the stored global-potential columns, so
any dataset (with or without an existing ``vcirc_kms``) can be re-gridded by rebuilding each row's
host potential (~18 ms/row) -- no stream re-simulation. Reuses ``recompute_vcirc`` from
``extend_vcirc_huang.py``; the legacy (pre-Ibata) potential is ``_host_potential``'s default.

Usage (CPU only -- importing hydrabflow.simulators pulls in JAX):
    JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0 python scripts/recompute_vcirc_grid.py IN.npz \
        --sim-yaml conf/simulator/stream_agama_ou24.yaml --out OUT.npz [--n-workers 32]
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from omegaconf import OmegaConf

from extend_vcirc_huang import recompute_vcirc  # noqa: E402  (same directory)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--sim-yaml", required=True, help="simulator yaml with obs_r_kpc/obs_vc_kms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-workers", type=int, default=32)
    args = ap.parse_args()

    sim = OmegaConf.load(args.sim_yaml).params
    r = np.asarray(sim.obs_r_kpc, dtype=float)
    vc_obs = np.asarray(sim.obs_vc_kms, dtype=float)

    raw = np.load(args.dataset, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    old = data.get("vcirc_kms")
    print(f"{args.dataset}: {len(data['j'])} rows; vcirc_kms "
          f"{None if old is None else old.shape} -> ({len(data['j'])}, {len(r)}, 1)")

    vc = recompute_vcirc(data, r, args.n_workers)
    data["vcirc_kms"] = vc[..., None].astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **data)
    src_hydra = os.path.join(os.path.dirname(os.path.abspath(args.dataset)), ".hydra")
    if os.path.isdir(src_hydra):
        shutil.copytree(src_hydra, os.path.splitext(args.out)[0] + ".hydra", dirs_exist_ok=True)

    n_nan = int(np.isnan(vc).any(axis=1).sum())
    med = np.nanmedian(vc, axis=0)
    print(f"Wrote {args.out}: {n_nan} NaN curves; dataset-median curve vs observed: "
          f"median |frac dev| = {np.median(np.abs(med - vc_obs) / vc_obs):.3f}")


if __name__ == "__main__":
    main()
