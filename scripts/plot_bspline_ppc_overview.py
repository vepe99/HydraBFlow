#!/usr/bin/env python
"""All-streams overview of one or more ``ppc_bspline_nn.py`` runs (streams x observables grid).

Reads each run's ``bspline_features.npz`` (the per-realization spline values on the grid, the real curve
and its bootstrap sigma), so the bands and z are exactly those of the runs' own ``ppc_<stream>.png`` /
``report.json``: z = (real - sim median) / sqrt(sigma_sim^2 + sigma_real^2), sigma_sim = 1.4826 MAD.
The real members are re-projected into the published STREAMFINDER frames for the scatter.

  .venv/bin/python scripts/plot_bspline_ppc_overview.py \
      --run "pooled global posterior=<dir>/aug_core080_pooled" --run "single-stream posterior=<dir>/aug_core080_stream" \
      --out <dir>/ppc_bspline_tracks.png
"""
import argparse
import os
import sys

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_observation_space import stream_frames, to_obs  # noqa: E402
from ppc_particle_coverage import real_clouds  # noqa: E402

NAMES = ["Pal5", "NGC3201", "M68"]
OBS = ["phi2 [deg]", "parallax [mas]", "mu_phi1 [mas/yr]", "mu_phi2 [mas/yr]", "v_los [km/s]"]
COLORS = ["#1f3a93", "#d9730d", "#2a7f3f", "#8e3b8e"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="LABEL=DIR of a ppc_bspline_nn.py output (repeatable)")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--simulator", default="stream_agama_spray_massloss_ibata_m200c_v4")
    ap.add_argument("--real-aug", default="stream_real_global")
    ap.add_argument("--title", default="B-spline posterior-predictive check (training augmentation stream_bspline_grid, core080)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [(r.split("=", 1)[0], np.load(os.path.join(r.split("=", 1)[1], "bspline_features.npz"))) for r in a.run]

    real = real_clouds(a.real, a.simulator, a.real_aug, 1000, 0)
    R_of = stream_frames("streamfinder", a.real)
    fig, axes = plt.subplots(len(NAMES), len(OBS), figsize=(21, 3.5 * len(NAMES)), squeeze=False)
    for s, name in enumerate(NAMES):
        Fr, r_vm = to_obs(R_of[s], real[s][0]), real[s][1]
        f0 = runs[0][1]
        for c, lab in enumerate(OBS, start=1):
            ax = axes[s, c - 1]
            gx = f0[f"{name}/grid_vlos"] if c == 5 else f0[f"{name}/grid"]
            rc, rs = f0[f"{name}/real_{c}"], f0[f"{name}/real_sigma_{c}"]
            title = []
            for k, (label, f) in enumerate(runs):
                col = COLORS[k % len(COLORS)]
                S = f[f"{name}/sim_{c}"]
                S = S[np.all(np.isfinite(S), 1)]
                if len(S) < 5:
                    continue
                q5, q16, q50, q84, q95 = np.percentile(S, [5, 16, 50, 84, 95], 0)
                ax.fill_between(gx, q5, q95, color=col, alpha=0.13, lw=0)
                ax.fill_between(gx, q16, q84, color=col, alpha=0.25, lw=0)
                ax.plot(gx, q50, color=col, lw=1.4, label=f"{label} ({len(S)} sims)")
                z = (rc - q50) / np.sqrt((1.4826 * np.median(np.abs(S - q50), 0)) ** 2 + rs ** 2)
                inside = np.mean((rc >= q5) & (rc <= q95))
                title.append(f"{label.split()[0]}: in {inside:.2f} z {np.nanmedian(z):+.2f}")
            sel = r_vm if c == 5 else np.ones(len(Fr), bool)
            ax.scatter(Fr[sel, 0], Fr[sel, c], s=5, c="0.35", alpha=0.45, lw=0)
            ax.fill_between(gx, rc - rs, rc + rs, color="k", alpha=0.18, lw=0, label="Gaia spline +- sigma (bootstrap)")
            ax.plot(gx, rc, color="k", lw=2.0, label="Gaia members (same spline)")
            ax.set_title("  ".join(title), fontsize=8)
            lo, hi = np.percentile(Fr[sel, c], [2, 98])
            pad = 0.6 * (hi - lo) + 1e-3
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_ylabel(lab)
            ax.grid(alpha=0.25)
            if c == 1:
                ax.text(0.02, 0.95, name, transform=ax.transAxes, fontweight="bold", va="top")
            if s == len(NAMES) - 1:
                ax.set_xlabel("phi1 [deg]  (STREAMFINDER frame)")
    axes[0, -1].legend(fontsize=7, loc="upper left")
    fig.suptitle(a.title, y=1.0)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
