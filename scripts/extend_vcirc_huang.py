"""Extend a stream dataset's rotation-curve observable onto larger radii (offline helper).

The stream simulators store every row's global potential parameters (halo ``rho/gamma/a/beta/q``,
disk ``r/z/Sigma``), and the model circular-velocity curve ``vcirc_kms`` is a *deterministic*
function of those parameters. So the model rotation curve at any radii — in particular the larger
Huang et al. (2016) radii (out to ~100 kpc, well beyond the Zhou 2023 range of ~24 kpc) — can be
recomputed from an existing dataset by rebuilding each potential (~18 ms/row), with **no** re-run
of the expensive stream integrator.

This script:
  1. loads a flat simulate dataset (with the global-potential columns + ``vcirc_kms``),
  2. rebuilds each row's host potential and re-evaluates the model rotation curve on the union
     grid ``stream_common.extended_rotation_curve()`` (Zhou below the split, Huang beyond),
  3. writes a copy of the dataset with ``vcirc_kms`` overwritten to ``(n, n_ext, 1)`` on that grid
     (plus traceability arrays ``vcirc_r_kpc`` / ``vcirc_obs_kms`` / ``vcirc_obs_sigma_kms``), and
  4. plots the predicted model curve (percentile band over the dataset) against the Zhou and Huang
     observations, and prints how the prediction at the appended large radii compares to Huang.

Usage:
    python scripts/extend_vcirc_huang.py DATASET.npz [--out COPY.npz] [--plot FIG.png]
        [--split-kpc 24.0] [--n-workers 16] [--r-min-kpc 5.5]

With no ``--out`` / ``--plot`` the copy lands next to the input as ``<stem>_huang.npz`` and the
figure as ``<stem>_huang_rotation_curve.png``.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hydrabflow.simulators.stream_agama import _agama, _host_potential, _vcirc  # noqa: E402
from hydrabflow.simulators.stream_common import (  # noqa: E402
    HUANG_R_KPC,
    HUANG_SIGMA_VC,
    HUANG_VC_KMS,
    OBS_R_KPC,
    OBS_SIGMA_VC,
    OBS_VC_KMS,
    extended_rotation_curve,
)

# Keys _host_potential reads to build the bulge+halo+disk host potential (bulge is fixed).
GLOBAL_KEYS = [
    "rho_TwoPowerTriaxial_halo", "gamma_TwoPowerTriaxial_halo", "a_TwoPowerTriaxial_halo",
    "beta_TwoPowerTriaxial_halo", "q_TwoPowerTriaxial_halo",
    "r_Disk", "z_Disk", "Sigma_Disk",
]


def _vcirc_worker(rows: list[dict], obs_r: np.ndarray) -> np.ndarray:
    """joblib worker: model rotation curve on ``obs_r`` for a chunk of parameter rows."""
    agama = _agama()
    agama.setNumThreads(1)
    out = np.full((len(rows), len(obs_r)), np.nan)
    for i, p in enumerate(rows):
        try:
            out[i] = _vcirc(_host_potential(agama, p), obs_r)
        except Exception:
            pass  # leave NaN; downstream masks/nan-aware stats handle it
    return out


def recompute_vcirc(data: dict[str, np.ndarray], obs_r: np.ndarray, n_workers: int) -> np.ndarray:
    from joblib import Parallel, delayed

    missing = [k for k in GLOBAL_KEYS if k not in data]
    if missing:
        raise KeyError(f"dataset is missing global-potential columns {missing}")
    n = len(np.asarray(data[GLOBAL_KEYS[0]]))
    rows = [{k: float(np.asarray(data[k]).reshape(n, -1)[i, 0]) for k in GLOBAL_KEYS}
            for i in range(n)]
    n_jobs = min(n_workers, n)
    chunks = [rows[i::n_jobs] for i in range(n_jobs)]
    res = Parallel(n_jobs=n_jobs)(delayed(_vcirc_worker)(c, obs_r) for c in chunks)
    vc = np.empty((n, len(obs_r)))
    for i, r in enumerate(res):
        vc[i::n_jobs] = r
    return vc


def plot_curve(vc: np.ndarray, r: np.ndarray, split: float, plot_path: str, r_min: float) -> None:
    pct = np.nanpercentile(vc, [5, 16, 50, 84, 95], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(r, pct[0], pct[4], color="C0", alpha=0.18, label="model 5-95%")
    ax.fill_between(r, pct[1], pct[3], color="C0", alpha=0.30, label="model 16-84%")
    ax.plot(r, pct[2], color="C0", lw=2, label="model median")
    ax.errorbar(OBS_R_KPC, OBS_VC_KMS, yerr=OBS_SIGMA_VC, fmt="o", ms=3, color="k", capsize=2,
                lw=1, label="Zhou+23 (training)")
    hi = HUANG_R_KPC > split
    ax.errorbar(HUANG_R_KPC[hi], HUANG_VC_KMS[hi], yerr=HUANG_SIGMA_VC[hi], fmt="s", ms=4,
                color="crimson", capsize=2, lw=1, label="Huang+16 (extended)")
    ax.errorbar(HUANG_R_KPC[~hi], HUANG_VC_KMS[~hi], yerr=HUANG_SIGMA_VC[~hi], fmt="s", ms=3,
                mfc="none", color="crimson", alpha=0.4, lw=0.8, label="Huang+16 (overlap)")
    ax.axvline(split, color="grey", ls="--", lw=1)
    ax.axvspan(r.min(), r_min, color="grey", alpha=0.10)
    ax.set_xscale("log")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel(r"$v_c$ [km/s]")
    ax.set_title(f"Model rotation curve extrapolated to Huang radii (n={len(vc)})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extend vcirc_kms onto Huang (2016) large radii.")
    ap.add_argument("dataset", help="flat simulate .npz with global-potential columns + vcirc_kms")
    ap.add_argument("--out", default=None, help="output copy .npz (default: <stem>_huang.npz)")
    ap.add_argument("--plot", default=None, help="figure path (default: <stem>_huang_rotation_curve.png)")
    ap.add_argument("--split-kpc", type=float, default=None,
                    help="keep Zhou below this radius, append Huang above (default: max Zhou radius)")
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--r-min-kpc", type=float, default=5.5, help="grey inner region on the plot")
    args = ap.parse_args()

    stem = os.path.splitext(args.dataset)[0]
    out_path = args.out or f"{stem}_huang.npz"
    plot_path = args.plot or f"{stem}_huang_rotation_curve.png"

    raw = np.load(args.dataset, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    r, vc_obs, sig_obs = extended_rotation_curve(args.split_kpc)
    split = float(OBS_R_KPC.max()) if args.split_kpc is None else float(args.split_kpc)
    n_orig = data["vcirc_kms"].shape[1] if "vcirc_kms" in data else 0
    print(f"Recomputing model rotation curve for {len(np.asarray(data[GLOBAL_KEYS[0]]))} rows "
          f"on {len(r)} radii ({n_orig} -> {len(r)}; {r.min():.2f}-{r.max():.2f} kpc) ...")
    vc = recompute_vcirc(data, r, args.n_workers)

    n_nan = int(np.isnan(vc).any(axis=1).sum())
    if n_nan:
        print(f"  {n_nan} rows have >=1 NaN on the extended grid (kept as NaN).")

    data["vcirc_kms"] = vc[..., None]  # (n, n_ext, 1)
    data["vcirc_r_kpc"] = r
    data["vcirc_obs_kms"] = vc_obs
    data["vcirc_obs_sigma_kms"] = sig_obs
    np.savez(out_path, **data)
    print(f"Wrote {out_path}  (vcirc_kms -> {data['vcirc_kms'].shape})")

    plot_curve(vc, r, split, plot_path, args.r_min_kpc)
    print(f"Wrote {plot_path}")

    # How does the prediction at the appended (Huang) radii compare to the Huang observations?
    ext = r > split
    med = np.nanmedian(vc, axis=0)
    lo, up = np.nanpercentile(vc, [5, 95], axis=0)
    inside = (vc_obs >= lo) & (vc_obs <= up)
    within1s = (vc_obs >= med - sig_obs) & (vc_obs <= med + sig_obs)  # obs within its own 1s of median
    summary = {
        "dataset": args.dataset,
        "out": out_path,
        "n_rows": int(vc.shape[0]),
        "n_radii": int(len(r)),
        "split_kpc": split,
        "n_extended_radii": int(ext.sum()),
        "extended_radii_kpc": r[ext].round(2).tolist(),
        "extended_median_fracdev_vs_huang": float(
            np.nanmedian(np.abs(med[ext] - vc_obs[ext]) / vc_obs[ext])
        ),
        "extended_obs_in_model_5_95_pct": float(inside[ext].mean()),
        "extended_obs_within_1sigma_of_model_median": float(within1s[ext].mean()),
        "n_rows_with_nan_on_extended_grid": n_nan,
    }
    print(json.dumps(summary, indent=2))
    with open(f"{stem}_huang_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
