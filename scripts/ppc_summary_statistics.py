#!/usr/bin/env python
"""Prior-predictive check of the stream-frame summary statistics: overlay simulated summary tracks
on the real Gaia streams.

Standalone (numpy + matplotlib only — no BayesFlow/JAX import, like ``scripts/ppc_rotation_curve.py``,
so it runs fast and on a GPU-less box). It fits the same per-stream data-driven great-circle frame
used by ``hydrabflow.augmentation.stream_summary`` (from the REAL members), projects both the real
members and a sample of simulated streams into ``(φ1, φ2, μ_φ1, μ_φ2)``, and overlays the binned
median±std tracks so you can confirm the summaries are realistic and localize which stream/quantity
is off.

The simulated draws come from a grouped multistream npz (``sim_data_projected`` ``(N,S,P,6)``, raw
model units, pre-observation-model); this script applies the per-stream observation window +
subsample to mimic the selection the training augmentation does, then projects with the real frame.

Usage:
  uv run python scripts/ppc_summary_statistics.py \
    --sim data/.../test_multistream_333.npz --out outputs/.../ppc_summary_statistics.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {0: "Pal5", 1: "NGC3201", 2: "M68"}
# Observation windows (ra_min, ra_max, dec_min, dec_max) + observed member counts, matching the
# reference stream configs (conf/augmentation/stream_global.yaml).
WINDOW = {
    0: (220.98008280733634, 250.0, -13.0, 10.0),
    1: (42.29847920913508, 140.0, -47.79545565626234, 20.812029301121846),
    2: (189.73694638139526, 280.0, -47.32109546202406, 70.0),
}
NOBS = {0: 129, 1: 195, 2: 297}
CH = dict(ra=0, dec=1, mu_ra=3, mu_dec=4, vlos=5)


def unit_vec(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def fit_frame(ra, dec):
    n = unit_vec(ra, dec)
    _, v = np.linalg.eigh(n.T @ n)
    pole = v[:, 0]
    mean = n.mean(0)
    mean = mean / np.linalg.norm(mean)
    x = mean - (mean @ pole) * pole
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(pole, x), pole], 0)


def project(R, ra, dec, mura, mudec):
    n = unit_vec(ra, dec)
    rar, decr = np.radians(ra), np.radians(dec)
    e = np.stack([-np.sin(rar), np.cos(rar), np.zeros_like(rar)], -1)
    m = np.stack([-np.sin(decr) * np.cos(rar), -np.sin(decr) * np.sin(rar), np.cos(decr)], -1)
    v = mura[:, None] * e + mudec[:, None] * m
    npr, vpr = n @ R.T, v @ R.T
    phi1 = np.degrees(np.arctan2(npr[:, 1], npr[:, 0]))
    phi2 = np.degrees(np.arcsin(np.clip(npr[:, 2], -1, 1)))
    p1, p2 = np.radians(phi1), np.radians(phi2)
    ep = np.stack([-np.sin(p1), np.cos(p1), np.zeros_like(p1)], -1)
    mp = np.stack([-np.sin(p2) * np.cos(p1), -np.sin(p2) * np.sin(p1), np.cos(p2)], -1)
    return phi1, phi2, np.sum(vpr * ep, 1), np.sum(vpr * mp, 1)


def window_subsample(s, j, rng):
    ra, dec = s[:, CH["ra"]], s[:, CH["dec"]]
    lo_ra, hi_ra, lo_dec, hi_dec = WINDOW[j]
    idx = np.where((ra >= lo_ra) & (ra <= hi_ra) & (dec >= lo_dec) & (dec <= hi_dec))[0]
    if len(idx) == 0:
        return None
    return s[rng.choice(idx, size=min(NOBS[j], len(idx)), replace=False)]


def binned_median(x, y, edges):
    return np.array(
        [
            np.median(y[(x >= lo) & (x <= hi)]) if ((x >= lo) & (x <= hi)).sum() else np.nan
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real",
        default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz",
        help="real observed-streams npz (defines the frames + real tracks)",
    )
    ap.add_argument("--sim", required=True, help="grouped multistream npz (N,S,P,6)")
    ap.add_argument("--out", default="ppc_summary_statistics.png")
    ap.add_argument("--n-sim", type=int, default=40)
    ap.add_argument("--k-track", type=int, default=10)
    ap.add_argument("--k-vlos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = np.load(args.real)
    rsim = d["sim_data_projected"]
    rsim = rsim[0] if rsim.ndim == 4 else rsim
    ram = d["attention_mask"]
    ram = ram[:, 0, :] if ram.ndim == 3 else ram
    rvm = d["vlos_mask"]
    rvm = rvm[:, 0, :] if rvm.ndim == 3 else rvm
    jarr = np.asarray(d["j"]).reshape(-1).astype(int)

    sd = np.load(args.sim)["sim_data_projected"]  # (N,S,P,6)

    labels = ["phi2 [deg]", "mu_phi1", "mu_phi2", "vlos"]
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for row in range(min(3, rsim.shape[0])):
        j = int(jarr[row])
        mem = ram[row].astype(bool)
        rs = rsim[row][mem]
        rvmask = (rvm[row].astype(bool) & mem)[mem]
        R = fit_frame(rs[:, CH["ra"]], rs[:, CH["dec"]])
        rphi1, rphi2, rm1, rm2 = project(R, rs[:, CH["ra"]], rs[:, CH["dec"]], rs[:, CH["mu_ra"]], rs[:, CH["mu_dec"]])
        rtracks = [rphi2, rm1, rm2, rs[:, CH["vlos"]]]

        te = np.quantile(rphi1, np.linspace(0, 1, args.k_track + 1))
        ve = np.quantile(rphi1[rvmask], np.linspace(0, 1, args.k_vlos + 1)) if rvmask.sum() > args.k_vlos else te
        tc, vc = 0.5 * (te[:-1] + te[1:]), 0.5 * (ve[:-1] + ve[1:])

        picks = rng.choice(sd.shape[0], size=min(args.n_sim, sd.shape[0]), replace=False)
        for p in picks:
            s = window_subsample(sd[p, row], j, rng)
            if s is None or len(s) < args.k_track:
                continue
            p1, p2, m1, m2 = project(R, s[:, CH["ra"]], s[:, CH["dec"]], s[:, CH["mu_ra"]], s[:, CH["mu_dec"]])
            strk = [p2, m1, m2, s[:, CH["vlos"]]]
            for c in range(4):
                edges, cen = (ve, vc) if c == 3 else (te, tc)
                axes[row, c].plot(cen, binned_median(p1, strk[c], edges), color="C0", alpha=0.25, lw=1)

        for c in range(4):
            ax = axes[row, c]
            edges, cen = (ve, vc) if c == 3 else (te, tc)
            sx = rphi1[rvmask] if c == 3 else rphi1
            sy = rtracks[c][rvmask] if c == 3 else rtracks[c]
            ax.scatter(sx, sy, s=6, alpha=0.35, color="C3", zorder=3)
            ax.plot(cen, binned_median(sx, sy, edges), "k-o", lw=2.5, ms=5, zorder=4)
            if row == 0:
                ax.set_title(labels[c])
            if c == 0:
                ax.set_ylabel(f"{NAMES.get(j, j)}")
            ax.set_xlabel("phi1 [deg]")

    fig.suptitle(
        f"Prior-predictive summary tracks: sim medians (blue) vs real Gaia (red pts, black median). "
        f"K_track={args.k_track}, K_vlos={args.k_vlos}",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
