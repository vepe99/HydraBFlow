"""Prior-predictive checks for the stellar-stream simulators (offline helper — not a Hydra stage).

Consumes a flat simulate dataset (or a single ``<dataset>.chunks/chunk_NNNNN.npz`` produced by
the resumable simulator) and asks whether the prior, pushed through the forward model, is
consistent with the real observations on two axes:

  (1) predicted rotation curve  -- percentile band of the model ``vcirc_kms`` curves vs the
      observed Zhou et al. (2023) curve (``stream_common.OBS_VC_KMS`` +/- ``OBS_SIGMA_VC``);
      reports the fraction of observed bins inside the prior 5-95% band and the prior-median
      fractional deviation over the r > ``r_min`` bins the network sees.
  (2) stream loci -- per target stream, the simulated particles' sky (RA/Dec) and proper-motion
      (mu_alpha*, mu_delta) distributions overlaid on the real Gaia members, plus the
      "real-locus reach": the fraction of real members with >=1 simulated particle within both a
      sky and a proper-motion tolerance. Reach ~1 means the training manifold covers the real
      stream; low reach flags a train-vs-reality gap (see the t_end PPC finding in CLAUDE.md).

Rows are grouped by the ``j`` stream index, so a flat training chunk (a mix of all target
streams) is the natural input. Real members come from the observed Gaia ``.npz`` (n=1 group,
m streams, ``attention_mask`` selecting valid particles).

Usage:
    python scripts/ppc_prior_predictive.py DATASET.npz OUT_DIR [--real PATH]
        [--sky-tol-deg 2.0] [--pm-tol-masyr 1.5] [--r-min-kpc 5.5]

Writes ``ppc_rotation_curve.png``, ``ppc_streams.png`` and ``ppc_summary.json`` to OUT_DIR and
prints the JSON summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from hydrabflow.simulators.stream_common import (  # noqa: E402
    OBS_R_KPC,
    OBS_SIGMA_VC,
    OBS_VC_KMS,
    extended_rotation_curve,
)

_REAL_NAME = "gaia_observed_streams_6Dwitherrors_cutNGC3201.npz"
# Prefer the in-repo copy (assets/gaia/, portable across clusters); fall back to the reference
# directory on the original machine. `copy the gaia data` populated assets/gaia/.
_REPO_REAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "assets", "gaia", _REAL_NAME)
_REF_REAL = ("/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/"
             "data/" + _REAL_NAME)
DEFAULT_REAL = _REPO_REAL if os.path.exists(_REPO_REAL) else _REF_REAL
# j index -> stream name (matches conf/simulator/stream_agama.yaml target_streams).
STREAM_NAMES = {0: "Pal5", 1: "NGC3201", 2: "M68"}
# sim_data_projected feature order (stream_common.sky_projection):
# (ra[deg], dec[deg], distance[kpc], pm_ra_cosdec[mas/yr], pm_dec[mas/yr], v_los[km/s]).


def load_dataset(path: str) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def real_members(real_path: str) -> dict[int, np.ndarray]:
    """Return {j: (n_valid, 6)} of the observed members, masked by ``attention_mask``."""
    d = np.load(real_path, allow_pickle=True)
    sp = d["sim_data_projected"][0]  # (m, n_particles, 6)
    am = d["attention_mask"]  # (m, 1, n_particles)
    js = d["j"].reshape(-1)  # (m,)
    return {int(j): sp[i][am[i, 0].astype(bool)] for i, j in enumerate(js)}


def rotation_curve_ppc(vc: np.ndarray, out_dir: str, r_min: float) -> dict:
    # Pick the observed reference to match the dataset's vcirc grid: the Zhou (34-radii) grid, or
    # the extended Zhou u Huang union grid when vcirc was generated with obs_r_grid=extended.
    n_r = vc.shape[1]
    if n_r == len(OBS_R_KPC):
        r, vc_ref, sig = OBS_R_KPC, OBS_VC_KMS, OBS_SIGMA_VC
        split = None
    else:
        r, vc_ref, sig = extended_rotation_curve()
        split = float(OBS_R_KPC.max())
        if len(r) != n_r:
            raise ValueError(f"vcirc has {n_r} radii, extended grid has {len(r)} - grid mismatch")

    pct = np.nanpercentile(vc, [5, 16, 50, 84, 95], axis=0)
    fig, ax = plt.subplots(figsize=(8.5 if split else 7.5, 5))
    ax.fill_between(r, pct[0], pct[4], color="C0", alpha=0.18, label="prior 5-95%")
    ax.fill_between(r, pct[1], pct[3], color="C0", alpha=0.30, label="prior 16-84%")
    ax.plot(r, pct[2], color="C0", lw=2, label="prior median")
    if split is None:
        ax.errorbar(r, vc_ref, yerr=sig, fmt="o", ms=3, color="k", capsize=2, lw=1,
                    label="Zhou+23 observed")
    else:
        lo = r <= split
        ax.errorbar(r[lo], vc_ref[lo], yerr=sig[lo], fmt="o", ms=3, color="k", capsize=2, lw=1,
                    label="Zhou+23 (<=24 kpc)")
        ax.errorbar(r[~lo], vc_ref[~lo], yerr=sig[~lo], fmt="s", ms=4, color="crimson", capsize=2,
                    lw=1, label="Huang+16 (>24 kpc)")
        ax.axvline(split, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
    ax.axvspan(r.min(), r_min, color="grey", alpha=0.12)
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel(r"$v_c$ [km/s]")
    ax.set_title(f"Prior-predictive rotation curve (n={len(vc)}, {n_r} radii)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "ppc_rotation_curve.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    use = r > r_min
    inside = (vc_ref >= pct[0]) & (vc_ref <= pct[4])
    med_fracdev = float(np.median(np.abs(pct[2][use] - vc_ref[use]) / vc_ref[use]))
    out = {
        "n_curves": int(len(vc)),
        "n_radii": int(n_r),
        "obs_bins_in_prior_5_95_pct": float(inside[use].mean()),
        "prior_median_vs_obs_median_fracdev": med_fracdev,
        "figure": path,
    }
    if split is not None:
        ext = r > split
        out["huang_bins_in_prior_5_95_pct"] = float(inside[ext].mean())
        out["prior_median_vs_huang_median_fracdev"] = float(
            np.median(np.abs(pct[2][ext] - vc_ref[ext]) / vc_ref[ext])
        )
    return out


def _reach(sim_xy, sim_pm, real_xy, real_pm, sky_tol, pm_tol) -> float:
    """Fraction of real members with >=1 sim particle within both sky and PM tolerance."""
    if len(sim_xy) == 0 or len(real_xy) == 0:
        return float("nan")
    tree = cKDTree(sim_xy)
    hit = 0
    for k in range(len(real_xy)):
        idx = tree.query_ball_point(real_xy[k], sky_tol)
        if idx and np.any(np.hypot(*(sim_pm[idx] - real_pm[k]).T) <= pm_tol):
            hit += 1
    return hit / len(real_xy)


def stream_ppc(chunk, out_dir, real_path, sky_tol, pm_tol) -> dict:
    sp = chunk["sim_data_projected"]  # (n, n_particles, 6)
    j = chunk["j"].reshape(-1).astype(int)
    real = real_members(real_path)
    present = [jj for jj in STREAM_NAMES if (j == jj).any()]
    metrics: dict = {}
    if not present:
        return metrics
    fig, axes = plt.subplots(len(present), 2, figsize=(11, 4.2 * len(present)), squeeze=False)
    for row, jj in enumerate(present):
        feats = sp[j == jj].reshape(-1, 6)
        feats = feats[np.isfinite(feats).all(axis=1)]  # drop NaN particles (invalid seeds)
        ra, dec, _, pmra, pmdec, _ = feats.T
        cosd = np.cos(np.deg2rad(dec))
        sim_xy = np.column_stack([ra * cosd, dec])
        sim_pm = np.column_stack([pmra, pmdec])
        rm = real.get(jj, np.empty((0, 6)))
        rra, rdec, rpmra, rpmdec = (rm[:, 0], rm[:, 1], rm[:, 3], rm[:, 4])
        real_xy = np.column_stack([rra * np.cos(np.deg2rad(rdec)), rdec]) if len(rm) else np.empty((0, 2))
        real_pm = np.column_stack([rpmra, rpmdec]) if len(rm) else np.empty((0, 2))

        ax = axes[row][0]
        ax.scatter(ra, dec, s=1, alpha=0.05, color="C0", rasterized=True, label="prior sim")
        if len(rm):
            ax.scatter(rra, rdec, s=4, color="crimson", label="real members")
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.set_title(f"{STREAM_NAMES[jj]} (j={jj}) sky  [n_sim_rows={(j == jj).sum()}]")
        ax.legend(fontsize=7, markerscale=3)

        ax = axes[row][1]
        ax.scatter(pmra, pmdec, s=1, alpha=0.05, color="C0", rasterized=True)
        if len(rm):
            ax.scatter(rpmra, rpmdec, s=4, color="crimson")
        ax.set_xlabel(r"$\mu_{\alpha}\cos\delta$ [mas/yr]")
        ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
        ax.set_title(f"{STREAM_NAMES[jj]} proper motion")

        metrics[STREAM_NAMES[jj]] = {
            "n_sim_rows": int((j == jj).sum()),
            "n_real_members": int(len(rm)),
            "real_locus_reach": _reach(sim_xy, sim_pm, real_xy, real_pm, sky_tol, pm_tol),
        }
    fig.suptitle(
        f"Prior-predictive streams vs real Gaia members "
        f"(reach tol: {sky_tol} deg sky, {pm_tol} mas/yr PM)", y=1.001
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "ppc_streams.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    metrics["figure"] = path
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Prior-predictive checks for stream simulators.")
    ap.add_argument("dataset", help="flat simulate dataset or a chunk .npz (with vcirc_kms + j)")
    ap.add_argument("out_dir", help="directory for figures + ppc_summary.json")
    ap.add_argument("--real", default=DEFAULT_REAL, help="observed Gaia streams .npz")
    ap.add_argument("--sky-tol-deg", type=float, default=2.0)
    ap.add_argument("--pm-tol-masyr", type=float, default=1.5)
    ap.add_argument("--r-min-kpc", type=float, default=5.5, help="vcirc cut floor (rnbody: 5.5)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    chunk = load_dataset(args.dataset)
    vc = chunk["vcirc_kms"].reshape(chunk["vcirc_kms"].shape[0], -1)
    summary = {
        "dataset": args.dataset,
        "n_rows": int(vc.shape[0]),
        "rotation_curve": rotation_curve_ppc(vc, args.out_dir, args.r_min_kpc),
        "streams": stream_ppc(chunk, args.out_dir, args.real, args.sky_tol_deg, args.pm_tol_masyr),
    }
    with open(os.path.join(args.out_dir, "ppc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
