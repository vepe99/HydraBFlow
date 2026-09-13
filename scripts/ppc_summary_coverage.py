#!/usr/bin/env python
"""Prior-predictive COVERAGE check of the per-bin stream summary statistics.

Question answered: does every observed (stream, statistic, phi1-bin) value — per-bin median AND
per-bin std of phi2 / mu_phi1 / mu_phi2 / v_los in the real-member phi1 bins — fall inside the
distribution the simulated prior sample produces for that same cell? If a cell of the real data
lies outside the prior-predictive range, no network trained on this prior can reproduce it and the
real posterior will be an extrapolation there.

Both sides go through the same estimator: the simulated groups are pushed through the TRAINING
observation model (window, member-count subsample, Gaia noise, v_los mask — ``augment_sim``) and
binned with the same quantile edges as the real members. Per cell we report the percentile rank of
the real value within the simulated distribution (0-100) and a robust z (vs median / 1.4826 MAD).

Outputs (in --out dir):
  summary_coverage_percentiles.png  heatmap: percentile per (stream x statistic) row, phi1-bin column
  summary_coverage_corner_<stream>.png  corner plot of a handful of headline scalars per stream
      (sim rows = grey, real = red lines) — the "does the observation sit inside the training set"
      picture at a glance
  summary_coverage.json  per-cell percentiles/z, and the fraction of cells inside the central 95 %.

Usage:
  uv run python scripts/ppc_summary_coverage.py --sim <multistream npz> --out <dir> \
      --aug stream_global_ibata_grid_v2 --simulator stream_agama_rnbody_ibata_m200c_v3
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppc_summary_statistics import (CH, NAMES, fit_frame, project, augment_sim, binned_median,
                                    binned_std)

QTY = ["phi2", "mu_phi1", "mu_phi2", "vlos"]
UNITS = {"phi2": "deg", "mu_phi1": "mas/yr", "mu_phi2": "mas/yr", "vlos": "km/s"}


def cell_stats(p1, tracks, vm, te, ve):
    """Per-bin median and std for the four quantities -> dict[(qty, stat)] = array over bins."""
    out = {}
    for c, q in enumerate(QTY):
        if q == "vlos":
            x, y, e = p1[vm], tracks[c][vm], ve
        else:
            x, y, e = p1, tracks[c], te
        out[(q, "med")] = binned_median(x, y, e)
        out[(q, "std")] = binned_std(x, y, e, min_count=3)
    return out


def percentile_rank(sim, real):
    s = sim[np.isfinite(sim)]
    if s.size < 10 or not np.isfinite(real):
        return np.nan, np.nan
    pct = 100.0 * np.mean(s <= real)
    mad = 1.4826 * np.median(np.abs(s - np.median(s)))
    z = (real - np.median(s)) / mad if mad > 0 else np.nan
    return pct, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="grouped multistream npz (N,S,P,6)")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--aug", default="stream_global_ibata_grid_v2")
    ap.add_argument("--simulator", default="stream_agama_rnbody_ibata_m200c_v3")
    ap.add_argument("--k-track", type=int, default=10)
    ap.add_argument("--k-vlos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.real)
    rsim = d["sim_data_projected"][0]; ram = d["attention_mask"][:, 0].astype(bool)
    rvm = d["vlos_mask"].astype(bool); jarr = d["j"].reshape(-1).astype(int)

    sd = np.load(args.sim)
    sdp = sd["sim_data_projected"]; jj = np.asarray(sd["j"]).reshape(sdp.shape[:2]).astype(int)
    sim, attn, vmask = augment_sim(sdp, jj, aug_preset=args.aug, simulator=args.simulator, seed=args.seed)
    n_groups = sdp.shape[0]

    report = {"sim": args.sim, "n_groups": int(n_groups), "streams": {}}
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for row in range(3):
        j = int(jarr[row]); name = NAMES[j]
        mem = ram[row]; rs = rsim[row][mem]; rvmask = (rvm[row] & mem)[mem]
        R = fit_frame(rs[:, CH["ra"]], rs[:, CH["dec"]])
        rphi1, rphi2, rm1, rm2 = project(R, rs[:, CH["ra"]], rs[:, CH["dec"]], rs[:, CH["mu_ra"]], rs[:, CH["mu_dec"]])
        te = np.quantile(rphi1, np.linspace(0, 1, args.k_track + 1))
        ve = np.quantile(rphi1[rvmask], np.linspace(0, 1, args.k_vlos + 1))
        real_cells = cell_stats(rphi1, [rphi2, rm1, rm2, rs[:, CH["vlos"]]], rvmask, te, ve)

        # simulated distribution per cell
        sim_cells = {k: [] for k in real_cells}
        n_used = 0
        for g in range(n_groups):
            s_idx = int(np.where(jj[g] == j)[0][0]) if (jj[g] == j).any() else row
            sel = attn[g, s_idx] & np.isfinite(sim[g, s_idx][:, CH["ra"]])
            if sel.sum() < args.k_track:
                continue
            s = sim[g, s_idx][sel]; svm = (vmask[g, s_idx] & attn[g, s_idx])[sel]
            p1, p2, m1, m2 = project(R, s[:, CH["ra"]], s[:, CH["dec"]], s[:, CH["mu_ra"]], s[:, CH["mu_dec"]])
            cells = cell_stats(p1, [p2, m1, m2, s[:, CH["vlos"]]], svm, te, ve)
            for k in sim_cells:
                sim_cells[k].append(cells[k])
            n_used += 1
        sim_cells = {k: np.array(v) for k, v in sim_cells.items()}  # (n_used, K)

        # percentiles / z per cell
        rows_lab, pct_mat, z_mat, cell_json = [], [], [], {}
        for q in QTY:
            for st in ("med", "std"):
                k = (q, st); K = real_cells[k].size
                pcts, zs = [], []
                for b in range(K):
                    pct, z = percentile_rank(sim_cells[k][:, b], real_cells[k][b])
                    pcts.append(pct); zs.append(z)
                pcts = np.array(pcts); zs = np.array(zs)
                rows_lab.append(f"{st} {q}")
                pct_mat.append(np.pad(pcts, (0, args.k_track - K), constant_values=np.nan))
                z_mat.append(np.pad(zs, (0, args.k_track - K), constant_values=np.nan))
                cell_json[f"{st}_{q}"] = dict(real=[float(x) for x in real_cells[k]],
                                              sim_p2_5=[float(x) for x in np.nanpercentile(sim_cells[k], 2.5, axis=0)],
                                              sim_p50=[float(x) for x in np.nanpercentile(sim_cells[k], 50, axis=0)],
                                              sim_p97_5=[float(x) for x in np.nanpercentile(sim_cells[k], 97.5, axis=0)],
                                              percentile=[float(x) for x in pcts], robust_z=[float(x) for x in zs])
        pct_mat = np.array(pct_mat); z_mat = np.array(z_mat)
        finite = np.isfinite(pct_mat)
        inside95 = ((pct_mat > 2.5) & (pct_mat < 97.5))[finite].mean()
        inside99 = ((pct_mat > 0.5) & (pct_mat < 99.5))[finite].mean()
        outside = [(rows_lab[i], int(b), float(pct_mat[i, b]), float(z_mat[i, b]))
                   for i in range(pct_mat.shape[0]) for b in range(pct_mat.shape[1])
                   if finite[i, b] and not (2.5 < pct_mat[i, b] < 97.5)]
        report["streams"][name] = dict(n_sim_used=int(n_used), n_real=int(mem.sum()), n_real_vlos=int(rvmask.sum()),
                                       frac_cells_inside_central95=float(inside95),
                                       frac_cells_inside_central99=float(inside99),
                                       cells_outside_central95=outside, cells=cell_json,
                                       phi1_edges_track=[float(x) for x in te], phi1_edges_vlos=[float(x) for x in ve])

        ax = axes[row]
        # distance from the median in "percentile units": 50 = perfect, 0/100 = outside
        im = ax.imshow(pct_mat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=100)
        ax.set_yticks(range(len(rows_lab))); ax.set_yticklabels(rows_lab, fontsize=8)
        ax.set_xticks(range(args.k_track)); ax.set_xticklabels([f"{0.5*(te[i]+te[i+1]):.0f}" for i in range(args.k_track)], fontsize=8)
        ax.set_xlabel("phi1 bin centre [deg] (v_los rows use their own 3 bins, left-aligned)")
        ax.set_title(f"{name}: percentile of the REAL value within the {n_used}-sim prior-predictive distribution "
                     f"(inside central 95 %: {100*inside95:.0f} % of cells; {len(outside)} outside)", fontsize=10)
        for i in range(pct_mat.shape[0]):
            for b in range(pct_mat.shape[1]):
                if finite[i, b]:
                    v = pct_mat[i, b]
                    ax.text(b, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if (v < 15 or v > 85) else "black",
                            fontweight="bold" if not (2.5 < v < 97.5) else "normal")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="percentile")

        # ---- corner of headline scalars: median-over-bins of each std, and the track medians at 3 bins
        mid = [1, args.k_track // 2, args.k_track - 2]
        scal_lab, sim_scal, real_scal = [], [], []
        for q in QTY:
            k = (q, "std")
            scal_lab.append(f"std {q}\n[{UNITS[q]}]"); sim_scal.append(np.nanmedian(sim_cells[k], axis=1)); real_scal.append(np.nanmedian(real_cells[k]))
        for q in ("phi2", "mu_phi1", "vlos"):
            k = (q, "med")
            bins = [0, 1, 2] if q == "vlos" else mid
            for b in bins:
                lo, hi = (ve if q == "vlos" else te)[b], (ve if q == "vlos" else te)[b + 1]
                scal_lab.append(f"med {q}\nphi1 {0.5*(lo+hi):.0f}"); sim_scal.append(sim_cells[k][:, b]); real_scal.append(real_cells[k][b])
        S = np.array(sim_scal).T; ok = np.isfinite(S).all(axis=1); S = S[ok]
        try:
            import corner
            figc = corner.corner(S, labels=scal_lab, truths=real_scal, truth_color="C3", color="0.35",
                                 bins=25, smooth=1.0, label_kwargs=dict(fontsize=8), quantiles=[0.025, 0.975],
                                 show_titles=False, plot_datapoints=True, data_kwargs=dict(alpha=0.25, ms=2))
            figc.suptitle(f"{name}: prior-predictive summary scalars from {S.shape[0]} sims (grey, dashed = central 95 %) "
                          f"vs the real Gaia value (red). {args.title}", fontsize=10, y=1.0)
            figc.savefig(os.path.join(args.out, f"summary_coverage_corner_{name}.png"), dpi=100, bbox_inches="tight")
            plt.close(figc)
        except Exception as e:  # corner missing or degenerate columns: keep the heatmap + JSON
            print("corner plot skipped:", e)

    fig.suptitle(f"Prior-predictive coverage of the per-bin summary statistics — {args.title}", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "summary_coverage_percentiles.png"), dpi=110); plt.close(fig)
    with open(os.path.join(args.out, "summary_coverage.json"), "w") as f:
        json.dump(report, f, indent=1)
    for name, r in report["streams"].items():
        print(f"{name:8s} sims {r['n_sim_used']:4d}  cells inside central 95%: {100*r['frac_cells_inside_central95']:.0f}%  "
              f"(99%: {100*r['frac_cells_inside_central99']:.0f}%)  outside:", [(a, b, round(p, 1)) for a, b, p, z in r["cells_outside_central95"]])


if __name__ == "__main__":
    main()
