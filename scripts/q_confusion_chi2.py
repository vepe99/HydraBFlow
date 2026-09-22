#!/usr/bin/env python
"""Where in the gridded summary input does the network read q_halo — and why would the real Pal5 look
PROLATE to it?

Builds the exact 14-channel ``sim_summary`` grid the 2-modal models ingest (training chain up to
``stream_summary_grid``) for N training rows and the real Gaia members (real chain), then per stream:

  1. predictability: a gradient-boosted regressor q <- grid cells, for feature subsets
     (all / statistics only / medians / dispersions / occupancy counts) -> which channels carry q at all;
  2. cell-wise Spearman(cell, q) map;
  3. chi^2 of every training row against the REAL grid, per channel (cells standardized by the sim
     spread), averaged in q bins -> which channels make prolate rows look closer to the real stream;
  4. ABC-style nearest neighbours: the q distribution of the k training rows closest to the real grid
     under each feature-subset distance -> the q a network conditioned on that subset would report;
  5. the occupancy mechanism: real per-bin counts vs the sim median counts by q tercile.

Usage:
  .venv/bin/python scripts/q_confusion_chi2.py --out outputs/v5_2modal/q_confusion --n-rows 30000 \
      [--scale std|mad] [--override augmentation.params.x=y]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_grid_coverage import NAMES, OCC, STAT, compose_aug, run_chain  # noqa: E402

CH = STAT + OCC  # 12 informative channels; 12 = j, 13 = phi1 centre (constant per stream)
SUBSETS = {
    "all (10 stats + 2 counts)": list(range(12)),
    "statistics only": list(range(10)),
    "medians only": [0, 2, 4, 6, 8],
    "dispersions only": [1, 3, 5, 7, 9],
    "occupancy only (n_track, n_vlos)": [10, 11],
    "medians + occupancy (nodisp layout)": [0, 2, 4, 6, 8, 10, 11],
}
Q = "q_TwoPowerTriaxial_halo"
Q_SD = 1.0 / np.sqrt(12.0)  # U[0.5,1.5]


def build_sim_grid(aug, flat, n_rows, seed, chunk=5000):
    steps = [str(s) for s in aug.steps]
    steps = steps[: steps.index("stream_summary_grid") + 1]
    steps = [s for s in steps if s not in ("add_noise_to_vcirc", "log10_vcirc")]
    out = []
    for lo in range(0, n_rows, chunk):
        hi = min(lo + chunk, n_rows)
        batch = {"sim_data_projected": np.asarray(flat["sim_data_projected"][lo:hi], np.float32),
                 "j": np.asarray(flat["j"][lo:hi], np.float32).reshape(-1, 1)}
        batch = run_chain(aug, steps, batch, seed + lo)
        out.append(np.asarray(batch["sim_summary"]))
        print(f"  sim grid rows {hi}/{n_rows}", flush=True)
    return np.concatenate(out)


def build_real_grid(raug, real_path, max_p):
    d = np.load(real_path)
    m = int(np.asarray(d["j"]).size)
    rb = {}
    for k in d.files:
        if k == "j":
            continue
        a = np.asarray(d[k], dtype=np.float32)
        a = a[:, :max_p] if a.ndim == 2 else a[:, :, :max_p]
        if a.ndim >= 3 and a.shape[0] == 1 and a.shape[1] == m:
            a = a.reshape(m, *a.shape[2:])
        if k in ("attention_mask", "vlos_mask") and a.ndim == 2:
            a = a[:, None, :]
        rb[k] = a
    rb["j"] = np.asarray(d["j"]).reshape(m, 1).astype(np.float32)
    rb = run_chain(raug, ["stream_summary_grid"], rb, 0)
    return np.asarray(rb["sim_summary"]), rb["j"].reshape(-1).astype(int)


def mask_undercount(G, min_count):
    """Stat cells of under-populated bins -> NaN (they are substituted zeros, not measurements)."""
    G = G.astype(float).copy()
    for c, lab in enumerate(STAT):
        occ = G[..., 11] if lab.endswith("vlos") else G[..., 10]
        G[..., c][occ < min_count] = np.nan
    return G


def predictability(X, q, subsets, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(q)); n_te = len(q) // 5
    te, tr = idx[:n_te], idx[n_te:]
    res = {}
    for name, chans in subsets.items():
        F = X[:, :, chans].reshape(len(q), -1)
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=seed).fit(F[tr], q[tr])
        p = m.predict(F[te])
        res[name] = dict(nrmse=float(np.sqrt(np.mean((p - q[te]) ** 2)) / Q_SD), corr=float(np.corrcoef(p, q[te])[0, 1]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_hydrabflow/training_data_100000.npz")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz")
    ap.add_argument("--simulator", default="stream_agama_spray_massloss_ibata_m200c_v4")
    ap.add_argument("--aug", default="stream_global_v5")
    ap.add_argument("--real-aug", default="stream_real_global_v5")
    ap.add_argument("--scale", default="std", choices=["std", "mad"], help="dispersion estimator (oldgrid = std, grid_v2 = mad)")
    ap.add_argument("--override", action="append", default=["simulator.params.n_particles=1000"])
    ap.add_argument("--n-rows", type=int, default=30000)
    ap.add_argument("--k-nn", type=int, default=300)
    ap.add_argument("--n-qbins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ov = list(args.override) + [f"augmentation.params.summary_scale={args.scale}"]

    cache = os.path.join(args.out, f"grids_{args.scale}_{args.n_rows}.npz")
    if os.path.exists(cache):
        z = np.load(cache); S, sj, q, R, rj = z["S"], z["sj"], z["q"], z["R"], z["rj"]; min_count = int(z["min_count"])
    else:
        aug = compose_aug(args.simulator, args.aug, ov)
        raug = compose_aug(args.simulator, args.real_aug, ov)
        min_count = int(aug.params.get("summary_min_count", 3))
        flat = np.load(args.data, mmap_mode="r")
        S = build_sim_grid(aug, flat, args.n_rows, args.seed)
        sj = np.asarray(flat["j"][: args.n_rows]).reshape(-1).astype(int)
        q = np.asarray(flat[Q][: args.n_rows]).reshape(-1)
        R, rj = build_real_grid(raug, args.real, int(aug.params.get("max_particles", 300)))
        np.savez(cache, S=S, sj=sj, q=q, R=R, rj=rj, min_count=min_count)
    K = S.shape[1]
    qedges = np.linspace(0.5, 1.5, args.n_qbins + 1)
    report = dict(scale=args.scale, n_rows=int(args.n_rows), k_nn=args.k_nn, streams={})

    for s, name in enumerate(NAMES):
        Gs = mask_undercount(S[sj == s], min_count)[:, :, :12]        # (n_s, K, 12)
        qs = q[sj == s]
        Gr = mask_undercount(R[rj == s][0][None], min_count)[0, :, :12]  # (K, 12)
        Gs = Gs[np.isfinite(qs)]; qs = qs[np.isfinite(qs)]
        rep = dict(n_sim=int(len(qs)))

        # 1. predictability
        rep["predictability"] = predictability(Gs, qs, SUBSETS, args.seed)
        print(f"\n== {name}  ({len(qs)} sims, scale={args.scale})  q predictability (test nRMSE / corr):")
        for k, v in rep["predictability"].items():
            print(f"   {k:38s} {v['nrmse']:.2f} / {v['corr']:.2f}")

        # 2. cell-wise Spearman with q
        rho = np.full((12, K), np.nan)
        for c in range(12):
            for b in range(K):
                v = Gs[:, b, c]; ok = np.isfinite(v)
                if ok.sum() > 50 and np.nanstd(v[ok]) > 0:
                    rho[c, b] = spearmanr(v[ok], qs[ok]).statistic

        # 3. chi^2 vs the real grid, cell-standardized by the sim spread
        med = np.nanmedian(Gs, 0); sd = 1.4826 * np.nanmedian(np.abs(Gs - med), 0)
        sd = np.where(sd > 0, sd, np.nanstd(Gs, 0)); sd = np.where(sd > 0, sd, 1.0)
        Z = (Gs - Gr) / sd                                              # (n, K, 12); NaN where either missing
        valid = np.isfinite(Z)
        z2 = np.where(valid, Z ** 2, 0.0)
        chi_ch = z2.sum(1) / np.maximum(valid.sum(1), 1)                # (n, 12) mean z^2 per channel
        n_valid_ch = valid.sum(1)
        qbin = np.clip(np.digitize(qs, qedges) - 1, 0, args.n_qbins - 1)
        chi_q = np.array([[np.nanmean(chi_ch[qbin == b, c]) if (n_valid_ch[qbin == b, c] > 0).any() else np.nan
                           for b in range(args.n_qbins)] for c in range(12)])  # (12, nq)
        z_real_pct = np.array([[100 * np.nanmean(Gs[:, b, c] <= Gr[b, c]) if np.isfinite(Gr[b, c]) else np.nan
                                for b in range(K)] for c in range(12)])

        # 4. ABC nearest neighbours per subset
        abc = {}
        for sub, chans in SUBSETS.items():
            d = z2[:, :, chans].sum((1, 2)) / np.maximum(valid[:, :, chans].sum((1, 2)), 1)
            d[valid[:, :, chans].sum((1, 2)) == 0] = np.inf
            nn = np.argsort(d)[: args.k_nn]
            abc[sub] = dict(q_median=float(np.median(qs[nn])), q_p16=float(np.percentile(qs[nn], 16)),
                            q_p84=float(np.percentile(qs[nn], 84)), frac_prolate=float(np.mean(qs[nn] > 1.0)),
                            dist_median=float(np.median(d[nn])), q_nn=qs[nn])
            print(f"   ABC {sub:38s} q = {abc[sub]['q_median']:.2f} [{abc[sub]['q_p16']:.2f},{abc[sub]['q_p84']:.2f}]  "
                  f"P(q>1) = {abc[sub]['frac_prolate']:.2f}   (dist {abc[sub]['dist_median']:.2f})")
        rep["abc"] = {k: {kk: vv for kk, vv in v.items() if kk != "q_nn"} for k, v in abc.items()}
        rep["chi2_by_qbin"] = dict(q_edges=qedges.tolist(), channels=CH, values=chi_q.tolist())
        rep["spearman_cells"] = rho.tolist()
        rep["real_percentile_cells"] = z_real_pct.tolist()
        # occupancy by q tercile
        terc = np.percentile(qs, [100 / 3, 200 / 3])
        occ_t = {lab: [np.nanmedian(Gs[qs < terc[0], :, c], 0), np.nanmedian(Gs[(qs >= terc[0]) & (qs < terc[1]), :, c], 0),
                       np.nanmedian(Gs[qs >= terc[1], :, c], 0)] for c, lab in ((10, "n_track"), (11, "n_vlos"))}
        rep["occupancy_by_q_tercile"] = {k: [a.tolist() for a in v] for k, v in occ_t.items()}
        rep["real_occupancy"] = dict(n_track=Gr[:, 10].tolist(), n_vlos=Gr[:, 11].tolist())
        report["streams"][name] = rep

        # ---- figure per stream
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        im = ax[0, 0].imshow(rho, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax[0, 0].set_yticks(range(12)); ax[0, 0].set_yticklabels(CH, fontsize=8); ax[0, 0].set_xlabel("phi1 bin")
        ax[0, 0].set_title("Spearman(cell, q_halo) over the training rows"); fig.colorbar(im, ax=ax[0, 0])
        for c in range(12):
            for b in range(K):
                if np.isfinite(rho[c, b]) and abs(rho[c, b]) > 0.1:
                    ax[0, 0].text(b, c, f"{rho[c,b]:+.2f}", ha="center", va="center", fontsize=6)
        im = ax[0, 1].imshow(np.log10(chi_q), aspect="auto", cmap="viridis")
        ax[0, 1].set_yticks(range(12)); ax[0, 1].set_yticklabels(CH, fontsize=8)
        ax[0, 1].set_xticks(range(args.n_qbins)); ax[0, 1].set_xticklabels([f"{a:.2f}-{b:.2f}" for a, b in zip(qedges[:-1], qedges[1:])], rotation=45, fontsize=7)
        ax[0, 1].set_title("log10 mean chi2 per channel of sims vs the REAL grid, by true q"); fig.colorbar(im, ax=ax[0, 1])
        for c in range(12):
            for b in range(args.n_qbins):
                if np.isfinite(chi_q[c, b]):
                    ax[0, 1].text(b, c, f"{chi_q[c,b]:.1f}", ha="center", va="center", fontsize=6, color="w")
        im = ax[0, 2].imshow(z_real_pct, aspect="auto", cmap="coolwarm", vmin=0, vmax=100)
        ax[0, 2].set_yticks(range(12)); ax[0, 2].set_yticklabels(CH, fontsize=8); ax[0, 2].set_xlabel("phi1 bin")
        ax[0, 2].set_title("percentile of the real cell in the sims"); fig.colorbar(im, ax=ax[0, 2])
        for c in range(12):
            for b in range(K):
                if np.isfinite(z_real_pct[c, b]):
                    ax[0, 2].text(b, c, f"{z_real_pct[c,b]:.0f}", ha="center", va="center", fontsize=6)
        bins = np.linspace(0.5, 1.5, 21)
        for sub, v in abc.items():
            ax[1, 0].hist(v["q_nn"], bins=bins, histtype="step", lw=1.5, label=f"{sub}: q={v['q_median']:.2f}, P(q>1)={v['frac_prolate']:.2f}")
        ax[1, 0].axvline(1.0, color="k", ls=":"); ax[1, 0].set_xlabel("q_halo of the k nearest training rows"); ax[1, 0].legend(fontsize=7)
        ax[1, 0].set_title(f"ABC: {args.k_nn} rows closest to the real {name} grid, per feature subset")
        for lab, col in (("n_track", 10), ("n_vlos", 11)):
            a = ax[1, 1] if lab == "n_track" else ax[1, 2]
            for t, (lbl, ls) in enumerate((("oblate tercile", "--"), ("middle", "-."), ("prolate tercile", "-"))):
                a.plot(range(K), occ_t[lab][t], ls=ls, color="C0", label=f"sim median, {lbl}")
            a.plot(range(K), Gr[:, col], "o-", color="red", label="real Gaia")
            a.set_xlabel("phi1 bin"); a.set_ylabel(lab); a.legend(fontsize=7); a.set_title(f"{lab}: stars per real-quantile phi1 bin")
        pr = rep["predictability"]
        fig.suptitle(f"{name} — q_halo in the gridded summary input (scale={args.scale}, {len(qs)} training rows).  "
                     "GBM q-predictability nRMSE: " + ", ".join(f"{k.split(' (')[0]} {v['nrmse']:.2f}" for k, v in pr.items()), fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"q_confusion_{name}_{args.scale}.png"), dpi=120); plt.close(fig)

    with open(os.path.join(args.out, f"q_confusion_{args.scale}.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
