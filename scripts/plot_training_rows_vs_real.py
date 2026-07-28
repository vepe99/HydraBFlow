#!/usr/bin/env python
"""Per-stream 10x10 grids: 100 individual TRAINING rows overlaid on the real Gaia members, each
panel also showing the bin approximation the summary statistics actually reduce the row to.

One PNG per stream (Pal5 / NGC3201 / M68). Each of the 100 panels is one row of the flat training
set, projected into that stream's data-driven great-circle frame (the same frame the summary
augmentation fits from the REAL members), with:

  * grey points  = real Gaia members (identical in every panel — the "true"/observed locus)
  * blue points  = this training row's stars
  * black line   = real binned median track  (the bin approximation of the observed stream)
  * blue line    = sim  binned median track +- per-bin std shading (what the network is fed)
  * dotted verticals = the phi1 bin edges (equal-count quantiles of the real members)

By default the sim rows are pushed through the TRAINING observation model (window -> member-count
subsample -> Gaia magnitudes -> DR3 errors -> noise -> v_los mask) via
``ppc_summary_statistics.augment_sim``, so a panel shows exactly what the summary network sees.
Use ``--no-noise`` for the raw noiseless streams.

Usage:
  uv run python scripts/plot_training_rows_vs_real.py \
    --sim data_local/data_agama_rnbody_ibata_m200c_v2_1e3/training_data_1000.npz \
    --out-dir data_local/data_agama_rnbody_ibata_m200c_v2_1e3/ppc/rows
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppc_summary_statistics import (  # noqa: E402  (same-dir import)
    CH,
    NAMES,
    augment_sim,
    binned_median,
    binned_std,
    fit_frame,
    project,
    window_subsample,
)

QTY = {
    "phi2": (0, "phi2 [deg]"),
    "mu_phi1": (1, "mu_phi1 [mas/yr]"),
    "mu_phi2": (2, "mu_phi2 [mas/yr]"),
    "vlos": (3, "vlos [km/s]"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="flat training npz (sim_data_projected (N,P,6))")
    ap.add_argument(
        "--real",
        default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz",
        help="real observed-streams npz (defines the frames + observed locus)",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=100, help="rows per stream (panels = n, grid 10x10)")
    ap.add_argument("--k-track", type=int, default=10)
    ap.add_argument("--k-vlos", type=int, default=3)
    ap.add_argument("--quantity", default="phi2", choices=sorted(QTY))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-noise", action="store_true", help="skip the training observation model")
    ap.add_argument("--aug", default="stream_global_ibata_grid_v2")
    ap.add_argument("--simulator", default="stream_agama_rnbody_ibata_m200c_v2")
    args = ap.parse_args()

    qi, qlabel = QTY[args.quantity]
    rng = np.random.default_rng(args.seed)

    d = np.load(args.sim)
    sd = d["sim_data_projected"]
    if sd.ndim == 4:  # grouped multistream npz: flatten members into rows
        n, s = sd.shape[:2]
        jflat = np.asarray(d["j"]).reshape(n, s)
        sd, jflat = sd.reshape(n * s, *sd.shape[2:]), jflat.reshape(-1)
    else:
        jflat = np.asarray(d["j"]).reshape(-1)
    jflat = jflat.astype(int)

    # --- pick the rows first, then augment only those (the observation model is per-row).
    picks = {}
    for j in sorted(set(jflat.tolist())):
        idx = np.where(jflat == j)[0]
        if len(idx) < args.n:
            print(f"stream {NAMES.get(j, j)}: only {len(idx)} rows available (< {args.n})")
        picks[j] = rng.choice(idx, size=min(args.n, len(idx)), replace=False)

    sel_all = np.concatenate([picks[j] for j in sorted(picks)])
    sub, jsub = sd[sel_all], jflat[sel_all]
    masks = None
    if not args.no_noise:
        sub, attn, vmask = augment_sim(
            sub[:, None, :, :], jsub[:, None], aug_preset=args.aug,
            simulator=args.simulator, seed=args.seed,
        )
        sub, masks = sub[:, 0], (attn[:, 0], vmask[:, 0])
    offset = {}
    o = 0
    for j in sorted(picks):
        offset[j] = o
        o += len(picks[j])

    real = np.load(args.real)
    rsim = real["sim_data_projected"]
    rsim = rsim[0] if rsim.ndim == 4 else rsim
    ram = real["attention_mask"]
    ram = ram[:, 0, :] if ram.ndim == 3 else ram
    rvm = real["vlos_mask"]
    rvm = rvm[:, 0, :] if rvm.ndim == 3 else rvm
    rj = np.asarray(real["j"]).reshape(-1).astype(int)

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for rrow in range(rsim.shape[0]):
        j = int(rj[rrow])
        if j not in picks:
            continue
        mem = ram[rrow].astype(bool)
        rs = rsim[rrow][mem]
        rvmask = (rvm[rrow].astype(bool) & mem)[mem]
        R = fit_frame(rs[:, CH["ra"]], rs[:, CH["dec"]])
        rp1, rp2, rm1, rm2 = project(
            R, rs[:, CH["ra"]], rs[:, CH["dec"]], rs[:, CH["mu_ra"]], rs[:, CH["mu_dec"]]
        )
        rtracks = [rp2, rm1, rm2, rs[:, CH["vlos"]]]

        te = np.quantile(rp1, np.linspace(0, 1, args.k_track + 1))
        ve = (
            np.quantile(rp1[rvmask], np.linspace(0, 1, args.k_vlos + 1))
            if rvmask.sum() > args.k_vlos else te
        )
        edges = ve if args.quantity == "vlos" else te
        cen = 0.5 * (edges[:-1] + edges[1:])
        rx = rp1[rvmask] if args.quantity == "vlos" else rp1
        ry = rtracks[qi][rvmask] if args.quantity == "vlos" else rtracks[qi]
        rmed = binned_median(rx, ry, edges)

        # shared axis ranges so all 100 panels are directly comparable
        xlo, xhi = np.min(rp1), np.max(rp1)
        pad_x = 0.08 * (xhi - xlo)
        ylo, yhi = np.percentile(ry, [1, 99])
        pad_y = 1.5 * (yhi - ylo) + 1e-6

        fig, axes = plt.subplots(10, 10, figsize=(34, 26), sharex=True, sharey=True)
        empty = 0
        for k, ax in enumerate(axes.ravel()):
            ax.tick_params(labelsize=6)
            ax.scatter(rx, ry, s=3, color="0.6", alpha=0.55, zorder=1)
            ax.plot(cen, rmed, "k-", lw=1.4, zorder=4)
            for e in edges:
                ax.axvline(e, color="0.85", ls=":", lw=0.6, zorder=0)
            if k >= len(picks[j]):
                ax.set_axis_off()
                continue
            p = offset[j] + k
            s = sub[p]
            if masks is not None:
                keep = masks[0][p] & np.isfinite(s[:, CH["ra"]])
                svm = masks[1][p] & masks[0][p]
            else:
                s2 = window_subsample(s, j, rng)
                keep = None
                if s2 is None:
                    empty += 1
                    ax.set_title(f"#{int(picks[j][k])} (empty)", fontsize=6, color="C3")
                    continue
                s, svm = s2, None
            if keep is not None:
                if keep.sum() < 3:
                    empty += 1
                    ax.set_title(f"#{int(picks[j][k])} (n={int(keep.sum())})", fontsize=6, color="C3")
                    continue
                svm = svm[keep]
                s = s[keep]
            p1, p2, m1, m2 = project(
                R, s[:, CH["ra"]], s[:, CH["dec"]], s[:, CH["mu_ra"]], s[:, CH["mu_dec"]]
            )
            strk = [p2, m1, m2, s[:, CH["vlos"]]]
            sx, sy = p1, strk[qi]
            if args.quantity == "vlos" and svm is not None:
                sx, sy = p1[svm], strk[qi][svm]
            ax.scatter(sx, sy, s=3, color="C0", alpha=0.6, zorder=2)
            smed = binned_median(sx, sy, edges)
            sstd = binned_std(sx, sy, edges)
            ax.plot(cen, smed, "-", color="C0", lw=1.4, zorder=5)
            ax.fill_between(cen, smed - sstd, smed + sstd, color="C0", alpha=0.22, zorder=3)
            ax.set_title(f"#{int(picks[j][k])}  n={len(sx)}", fontsize=6)

        for ax in axes[-1]:
            ax.set_xlabel("phi1 [deg]", fontsize=7)
        for ax in axes[:, 0]:
            ax.set_ylabel(qlabel, fontsize=7)
        axes[0, 0].set_xlim(xlo - pad_x, xhi + pad_x)
        axes[0, 0].set_ylim(ylo - pad_y, yhi + pad_y)
        noise = "raw (noiseless)" if args.no_noise else "noise-convolved (training observation model)"
        fig.suptitle(
            f"{NAMES.get(j, j)}: {len(picks[j])} training rows [{noise}] vs real Gaia members "
            f"(grey pts / black median). Blue = row stars, blue line +- band = binned median +- "
            f"per-bin std; dotted = phi1 bin edges (K={len(cen)}). {empty} panel(s) had too few "
            "stars in the window.",
            fontsize=15,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        out = os.path.join(
            args.out_dir, f"training_rows_{NAMES.get(j, j)}_{args.quantity}.png"
        )
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print("saved", out, f"({empty} empty panels)")
        written.append(out)

    print("\n".join(written))


if __name__ == "__main__":
    main()
