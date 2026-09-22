#!/usr/bin/env python
"""Rank stripping ages: how well does each t_end reproduce the REAL streams?

Reads a set of single-realization runs of `stream_agama_rnbody_mcmillan17_mean` that differ only in
`t_end` (see the 2026-09-22 session entry in CLAUDE.md) and scores each stream at each age on three
model-free statistics, all computed in that stream's real-fitted great-circle frame with equal-count
phi1 bins of the REAL members:

  track   median over bins of |median phi2 (sim) - median phi2 (real)|, in deg. Lower is better.
  width   median over bins of the robust dispersion (1.4826 x MAD) of phi2, as a RATIO sim/real.
          1.0 is right; < 1 means the simulated stream is too thin, > 1 too wide.
  edge    the phi1 number-density ratio (mean of the two outer bins / mean of the inner bins).
          Real streams fall off towards their ends; a simulated stream that overflows its window
          piles up at the edges instead (see the 2026-07-29 entry: for the wide-window streams this,
          NOT the in-window phi1 extent, is the arm-length statistic, because the extent saturates).
          Computed on the NOISELESS in-window particles, so it is not sample noise. NOTE the bins
          are equal-count quantiles of the REAL members, so the real stream's own ratio is 1.0 BY
          CONSTRUCTION — it is a reference value, not a measurement. The comparison is still
          meaningful (it asks whether the sim puts the same share of stars in the outer quantile
          bins as the data does), but do not read `edge_real` as an observed property.

Also reported, because they decide whether a statistic means anything at that age:
  n_win   stored in-window particles (the observation model draws its members from these)
  bound   m_bound_final / m_progenitor — the real clusters all still exist, so 0 is a failure
  clump   fraction of in-window stars within 1 deg (in phi1) of the peak-density bin = the
          surviving remnant. A large value means the "stream" is mostly cluster members and the
          width/track numbers describe the progenitor, not the tails.

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/scan_tend_metrics.py \
    --runs data_local/mcmillan17_scan/t*/mcmillan17_mean.npz \
           data_local/mcmillan17_mean_tend3/mcmillan17_mean.npz \
           data_local/mcmillan17_mean/mcmillan17_mean.npz \
    --out data_local/mcmillan17_scan
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

from ppc_summary_statistics import (  # noqa: E402  (same-dir import)
    NAMES,
    WINDOW,
    augment_sim,
    binned_mad_std,
    binned_median,
    fit_frame,
    project,
)

DEFAULT_REAL = "assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz"
COLOR = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}


def load_real(path, k_bins):
    d = np.load(path)
    rs = d["sim_data_projected"]
    rs = rs[0] if rs.ndim == 4 else rs
    am = d["attention_mask"]
    am = am[:, 0, :] if am.ndim == 3 else am
    jj = np.asarray(d["j"]).reshape(-1).astype(int)
    frames, proj, edges = {}, {}, {}
    for i in range(rs.shape[0]):
        j = int(jj[i])
        m = rs[i][am[i].astype(bool)]
        frames[j] = fit_frame(m[:, 0], m[:, 1])
        p1, p2, _, _ = project(frames[j], m[:, 0], m[:, 1], m[:, 3], m[:, 4])
        proj[j] = (p1, p2)
        edges[j] = np.quantile(p1, np.linspace(0, 1, k_bins + 1))
    return frames, proj, edges


def edge_centre(p1, edges):
    n = np.array([((p1 >= a) & (p1 <= b)).sum() for a, b in zip(edges[:-1], edges[1:])], float)
    return float((n[0] + n[-1]) / 2 / max(n[1:-1].mean(), 1e-9))


def score_run(npz, frames, proj, edges, aug_preset, seed):
    d = np.load(npz)
    sd = d["sim_data_projected"]
    jj = np.asarray(d["j"]).reshape(1, -1).astype(int)
    aug, amask, _ = augment_sim(sd, jj, aug_preset=aug_preset, seed=seed)
    rows = []
    for s in range(sd.shape[1]):
        j = int(jj[0, s])
        rp1, rp2 = proj[j]
        e = edges[j]
        st = aug[0, s][amask[0, s].astype(bool)]
        p1, p2, _, _ = project(frames[j], st[:, 0], st[:, 1], st[:, 3], st[:, 4])
        track = float(np.nanmedian(np.abs(binned_median(p1, p2, e) - binned_median(rp1, rp2, e))))
        ws = float(np.nanmedian(binned_mad_std(p1, p2, e)))
        wr = float(np.nanmedian(binned_mad_std(rp1, rp2, e)))
        raw = sd[0, s]
        lo, hi, dlo, dhi = WINDOW[j]
        w = (raw[:, 0] >= lo) & (raw[:, 0] <= hi) & (raw[:, 1] >= dlo) & (raw[:, 1] <= dhi)
        q1, _, _, _ = project(frames[j], raw[w, 0], raw[w, 1], raw[w, 3], raw[w, 4])
        hist, be = np.histogram(q1, bins=60)
        peak = 0.5 * (be[:-1] + be[1:])[hist.argmax()]
        m0 = float(d["m_progenitor"].reshape(-1)[s])
        rows.append(dict(
            stream=NAMES[j], j=j, t_end=float(d["t_end"].reshape(-1)[s]),
            track=track, width_ratio=ws / wr, edge=edge_centre(q1, e),
            edge_real=edge_centre(rp1, e), n_win=int(w.sum()),
            bound_frac=float(d["m_bound_final"].reshape(-1)[s]) / m0,
            clump=float(np.mean(np.abs(q1 - peak) < 1.0)),
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--real", default=DEFAULT_REAL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aug-preset", default="stream_global_v5")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames, proj, edges = load_real(args.real, args.bins)
    rows = []
    for npz in args.runs:
        rows += score_run(npz, frames, proj, edges, args.aug_preset, args.seed)
    rows.sort(key=lambda r: (r["j"], r["t_end"]))

    hdr = (f"{'stream':9s} {'t_end':>5s} {'n_win':>6s} {'bound':>6s} {'clump':>6s} "
           f"{'track':>7s} {'width':>7s} {'edge':>6s} {'(real)':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['stream']:9s} {r['t_end']:5.1f} {r['n_win']:6d} {r['bound_frac']:6.2f} "
              f"{r['clump']:6.2f} {r['track']:7.3f} {r['width_ratio']:7.2f} {r['edge']:6.2f} "
              f"{r['edge_real']:7.2f}")
    with open(os.path.join(args.out, "scan_metrics.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    metrics = [("track", "median |d phi2| [deg]  (0 = perfect)", None),
               ("width_ratio", "robust width  sim / real", 1.0),
               ("edge", "phi1 edge/centre density", None),
               ("bound_frac", "m_bound_final / m_progenitor", None),
               ("clump", "in-window fraction within 1 deg of remnant", None),
               ("n_win", "stored in-window particles", None)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (key, lab, ref) in zip(axes.ravel(), metrics):
        for j in sorted(NAMES):
            sel = sorted([r for r in rows if r["j"] == j], key=lambda r: r["t_end"])
            ax.plot([r["t_end"] for r in sel], [r[key] for r in sel], "o-", c=COLOR[j],
                    label=NAMES[j])
            if key == "edge":
                ax.axhline(sel[0]["edge_real"], ls=":", c=COLOR[j], lw=1)
        if ref is not None:
            ax.axhline(ref, ls="--", c="k", lw=1)
        if key == "n_win":
            ax.set_yscale("log")
        ax.set_xlabel("t_end [Gyr]")
        ax.set_title(lab, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Stripping-age scan, McMillan (2017) fixed potential, all local parameters at "
                 "their prior mean (dotted = the real streams' edge/centre)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "scan_tend_metrics.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
