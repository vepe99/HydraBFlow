"""Observation-space coverage of the real members by posterior-predictive (or prior) star clouds.

Per stream: the real Gaia members as points in CATALOGUE coordinates (dec, parallax, mu_alpha*,
mu_delta, v_los against RA — no stream frame) with the noise-convolved simulated clouds' 5-95 %
and 16-84 % bands per RA bin drawn over them. Reports the fraction of real stars inside the 5-95 %
band per observable ("do we cover the stream?"). Sims go through the TRAINING observation model
(`augment_sim`, up to mask_vlos), so sim and real carry the same errors; v_los uses measured
stars only on both sides.

Usage:
  python scripts/ppc_icrs_coverage.py --sim <run>/ppc_posterior_streams_pooled.npz \
      --real assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz \
      --simulator stream_agama_spray_massloss_ibata_m200c_v4 --aug stream_global --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import augment_sim  # noqa: E402

COLS = ["dec [deg]", "parallax [mas]", "mu_alpha* [mas/yr]", "mu_delta [mas/yr]", "v_los [km/s]"]
STREAMS = {0: "Pal5", 1: "NGC3201", 2: "M68"}
QS = [0.05, 0.16, 0.5, 0.84, 0.95]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", required=True, help="grouped npz (N,S,P,6) with j (N,S)")
    ap.add_argument("--real", required=True)
    ap.add_argument("--simulator", required=True)
    ap.add_argument("--aug", default="stream_global")
    ap.add_argument("--bins", type=int, default=12, help="equal-count RA bins of the real members")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.sim)
    sd, j = d["sim_data_projected"], d["j"].reshape(d["sim_data_projected"].shape[0], -1)
    sim, attn, vmask = augment_sim(sd, j, aug_preset=args.aug, simulator=args.simulator, seed=args.seed)
    attn, vmask = attn.astype(bool), vmask.astype(bool)

    r = np.load(args.real)
    rx = r["sim_data_projected"][0]                      # (S, P, 6)
    ratt = r["attention_mask"].reshape(rx.shape[0], -1).astype(bool)
    rvm = r["vlos_mask"].reshape(rx.shape[0], -1).astype(bool)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = rx.shape[0]
    fig, axes = plt.subplots(len(COLS), S, figsize=(5.2 * S, 2.6 * len(COLS)), squeeze=False)
    report = {}
    for s in range(S):
        name = STREAMS.get(s, f"stream{s}")
        js = int(np.round(np.median(j[:, s])))
        real = rx[s][ratt[s]]
        real_vm = rvm[s][ratt[s]]
        # pool every realization's attended stars of this stream
        sel = (j == js) & True
        cloud = sim[sel][attn[sel]]                       # (n_stars, 6)
        cloud_vm = vmask[sel][attn[sel]]
        edges = np.quantile(real[:, 0], np.linspace(0, 1, args.bins + 1))
        ctr = 0.5 * (edges[1:] + edges[:-1])
        rep = {}
        for c, lbl in enumerate(COLS, start=1):
            ax = axes[c - 1, s]
            keep_s = cloud_vm if c == 5 else np.ones(len(cloud), bool)
            keep_r = real_vm if c == 5 else np.ones(len(real), bool)
            cs, rs = cloud[keep_s], real[keep_r]
            q = np.full((args.bins, len(QS)), np.nan)
            for b in range(args.bins):
                m = (cs[:, 0] >= edges[b]) & (cs[:, 0] <= edges[b + 1]) & np.isfinite(cs[:, c])
                if m.sum() >= 5:
                    q[b] = np.quantile(cs[m, c], QS)
            ok = np.isfinite(q[:, 0])
            ax.fill_between(ctr[ok], q[ok, 0], q[ok, 4], color="C0", alpha=0.18, lw=0, label="sim 5-95 %")
            ax.fill_between(ctr[ok], q[ok, 1], q[ok, 3], color="C0", alpha=0.35, lw=0, label="sim 16-84 %")
            ax.plot(ctr[ok], q[ok, 2], color="C0", lw=1.2, label="sim median")
            ax.scatter(rs[:, 0], rs[:, c], s=9, color="k", zorder=5, label="real members")
            # coverage: real star inside its bin's 5-95 band
            b_of = np.clip(np.searchsorted(edges, rs[:, 0], side="right") - 1, 0, args.bins - 1)
            lo, hi = q[b_of, 0], q[b_of, 4]
            inside = (rs[:, c] >= lo) & (rs[:, c] <= hi)
            valid = np.isfinite(lo)
            frac = float(inside[valid].mean()) if valid.any() else float("nan")
            rep[lbl] = {"frac_real_inside_5_95": frac, "n_real": int(valid.sum()),
                        "n_sim_stars": int(np.isfinite(cs[:, c]).sum())}
            ax.set_title(f"{name}: {frac:.0%} of real inside 5-95 %", fontsize=9)
            ax.set_ylabel(lbl)
            lo_y, hi_y = np.nanmin(np.r_[q[ok, 0], rs[:, c]]), np.nanmax(np.r_[q[ok, 4], rs[:, c]])
            pad = 0.08 * (hi_y - lo_y)
            ax.set_ylim(lo_y - pad, hi_y + pad)
            if c == len(COLS):
                ax.set_xlabel("RA [deg]")
        report[name] = rep
        axes[0, s].legend(fontsize=7, loc="best")
    # observation-space corner per stream: PPC cloud (68/95 % contours) with the real members on top
    import corner
    ALL = ["RA [deg]"] + COLS
    for s in range(S):
        name = STREAMS.get(s, f"stream{s}")
        js = int(np.round(np.median(j[:, s])))
        sel = j == js
        cloud, cvm = sim[sel][attn[sel]], vmask[sel][attn[sel]]
        real, rvm_s = rx[s][ratt[s]], rvm[s][ratt[s]]
        rng = np.random.default_rng(args.seed)
        fin = np.isfinite(cloud).all(1)
        for tag, cols, cs, rs in [
            ("astrometry", slice(0, 5), cloud[fin][:, :5], real[:, :5]),          # ALL members
            ("6d", slice(0, 6), cloud[fin & cvm], real[rvm_s]),                  # measured v_los only
        ]:
            if len(cs) > 20000:
                cs = cs[rng.choice(len(cs), 20000, replace=False)]
            labels = ALL[cols]
            rngs = [(min(a.min(), b.min()), max(a.max(), b.max())) for a, b in
                    zip(np.quantile(cs, [0.005, 0.995], axis=0).T, np.quantile(rs, [0.0, 1.0], axis=0).T)]
            rngs = [(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)) for lo, hi in rngs]
            fg = corner.corner(cs, labels=labels, range=rngs, color="C0", levels=(0.68, 0.95),
                               plot_datapoints=False, plot_density=False, fill_contours=True,
                               hist_kwargs={"density": True}, smooth=1.0)
            # real members ON TOP of the filled contours (corner puts datapoints at zorder -1)
            k = len(labels)
            axs = np.array(fg.axes).reshape(k, k)
            for a in range(k):
                for b in range(a):
                    axs[a, b].scatter(rs[:, b], rs[:, a], s=6, color="k", alpha=0.8, zorder=10)
                axs[a, a].hist(rs[:, a], bins=20, range=rngs[a], density=True, histtype="step",
                               color="k", lw=1.2, zorder=10)
            what = "all members" if tag == "astrometry" else "measured v_los only"
            fg.suptitle(f"{name}: observation-space corner ({tag}) — PPC cloud (blue 68/95 %) vs real "
                        f"members (black, {what}, n={len(rs)})", y=1.01)
            fg.savefig(os.path.join(args.out, f"corner_{name}_{tag}.png"), dpi=110, bbox_inches="tight")
            plt.close(fg)

    fig.suptitle(args.title or f"Observation-space PPC: {os.path.basename(args.sim)} vs real members "
                 f"({sim.shape[0]} realizations, training observation model)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ppc_icrs_coverage.png"), dpi=130)
    json.dump(report, open(os.path.join(args.out, "coverage.json"), "w"), indent=2)
    for name, rep in report.items():
        print(name, {k: round(v["frac_real_inside_5_95"], 2) for k, v in rep.items()})


if __name__ == "__main__":
    main()
