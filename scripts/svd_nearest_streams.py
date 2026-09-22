#!/usr/bin/env python
"""SVD of the gridded summary input per stream; project the REAL stream onto the first n PCs and find
the nearest training rows there (an ABC posterior in PC space) + which channels load on those PCs.
Reads the grid cache written by q_confusion_chi2.py."""
import argparse, json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
STAT = ["med_phi2", "std_phi2", "med_plx", "std_plx", "med_mu_phi1", "std_mu_phi1", "med_mu_phi2", "std_mu_phi2", "med_vlos", "std_vlos"]
CH = STAT + ["n_track", "n_vlos"]; NAMES = ["Pal5", "NGC3201", "M68"]

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="outputs/v5_2modal/q_confusion/grids_std_30000.npz")
ap.add_argument("--n-pc", type=int, default=10); ap.add_argument("--k", type=int, default=300)
ap.add_argument("--out", default="outputs/v5_2modal/q_confusion")
ap.add_argument("--no-occupancy", action="store_true", help="statistics only (what the masked backbone feeds the transformer)")
a = ap.parse_args()
z = np.load(a.cache); S, sj, q, R, rj, min_count = z["S"], z["sj"], z["q"], z["R"], z["rj"], int(z["min_count"])
K = S.shape[1]; rep = {}
fig, axes = plt.subplots(3, 3, figsize=(20, 13))
for s, name in enumerate(NAMES):
    G = S[sj == s][:, :, :12].astype(float); qs = q[sj == s]; ok = np.isfinite(qs); G, qs = G[ok], qs[ok]
    Gr = R[rj == s][0][:, :12].astype(float)
    # under-populated stat cells are substituted zeros -> treat as missing, fill with the training mean
    for c, lab in enumerate(STAT):
        occ = 11 if lab.endswith("vlos") else 10
        G[..., c][G[..., occ] < min_count] = np.nan
        Gr[:, c][Gr[:, occ] < min_count] = np.nan
    nch = 10 if a.no_occupancy else 12
    X = G[..., :nch].reshape(len(qs), -1); x_real = Gr[:, :nch].reshape(-1)
    mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[~(sd > 0)] = 1.0
    Xz = np.where(np.isfinite(X), (X - mu) / sd, 0.0); xr = np.where(np.isfinite(x_real), (x_real - mu) / sd, 0.0)
    U, sv, Vt = np.linalg.svd(Xz, full_matrices=False)
    P = Xz @ Vt[: a.n_pc].T; pr = Vt[: a.n_pc] @ xr           # scores
    var = sv**2 / (sv**2).sum()
    d = np.sqrt(((P - pr) ** 2).sum(1)); nn = np.argsort(d)[: a.k]
    # how typical is the real point per PC (percentile) and its Mahalanobis in PC space
    pc_pct = [100 * np.mean(P[:, i] <= pr[i]) for i in range(a.n_pc)]
    maha = np.sqrt((((pr - P.mean(0)) / P.std(0)) ** 2).sum()); maha_sims = np.sqrt((((P - P.mean(0)) / P.std(0)) ** 2).sum(1))
    # PC loadings by channel (sum of squared loadings over bins) and PC-q correlation
    load = (Vt[: a.n_pc] ** 2).reshape(a.n_pc, K, nch).sum(1)
    rho_q = [np.corrcoef(P[:, i], qs)[0, 1] for i in range(a.n_pc)]
    # a linear q-readout in PC space: how much q the first n PCs carry, and what it says for the real point
    A = np.c_[P, np.ones(len(qs))]; coef, *_ = np.linalg.lstsq(A, qs, rcond=None); q_lin = A @ coef
    r2 = 1 - np.mean((q_lin - qs) ** 2) / qs.var(); q_real_lin = float(np.r_[pr, 1] @ coef)
    rep[name] = dict(var_explained_first_n=float(var[: a.n_pc].sum()), pc_percentile_of_real=pc_pct,
                     mahalanobis_real=float(maha), mahalanobis_real_pct=float(100 * np.mean(maha_sims <= maha)),
                     nn_q_median=float(np.median(qs[nn])), nn_q_p16=float(np.percentile(qs[nn], 16)), nn_q_p84=float(np.percentile(qs[nn], 84)),
                     nn_frac_prolate=float(np.mean(qs[nn] > 1)), nn_dist_median=float(np.median(d[nn])),
                     sims_median_nn_dist=float(np.median(np.sort(((P[None, :500] - P[:500, None]) ** 2).sum(-1) ** 0.5, 1)[:, 1:a.k // 10 + 1])),
                     linear_q_readout_r2=float(r2), linear_q_readout_real=q_real_lin,
                     pc_corr_with_q=[float(r) for r in rho_q],
                     pc_top_channels=[[CH[j] for j in np.argsort(load[i])[::-1][:3]] for i in range(a.n_pc)])
    print(f"\n== {name}: first {a.n_pc} PCs explain {100*var[:a.n_pc].sum():.0f}% of the cell variance")
    for i in range(a.n_pc):
        print(f"  PC{i+1:2d} var {100*var[i]:4.1f}%  corr(q) {rho_q[i]:+.2f}  real at {pc_pct[i]:5.1f} pct   loads: " + ", ".join(f"{CH[j]} {load[i,j]:.2f}" for j in np.argsort(load[i])[::-1][:3]))
    print(f"  real Mahalanobis in PC space {maha:.1f} (at the {rep[name]['mahalanobis_real_pct']:.1f} pct of the sims)")
    print(f"  {a.k} nearest training rows: q = {rep[name]['nn_q_median']:.2f} [{rep[name]['nn_q_p16']:.2f},{rep[name]['nn_q_p84']:.2f}]  P(q>1) = {rep[name]['nn_frac_prolate']:.2f}; their distance {rep[name]['nn_dist_median']:.2f} vs typical sim-to-sim {rep[name]['sims_median_nn_dist']:.2f}")
    print(f"  linear q readout from these PCs: R^2 {r2:.2f} on sims; applied to the real point -> q = {q_real_lin:.2f}")
    ax = axes[s]
    ax[0].scatter(P[:, 0], P[:, 1], c=qs, s=3, cmap="coolwarm", vmin=0.5, vmax=1.5); ax[0].plot(pr[0], pr[1], "k*", ms=16, mec="w"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].set_title(f"{name}: sims coloured by q, star = real")
    ax[1].hist(qs, bins=20, range=(0.5, 1.5), histtype="step", color="grey", density=True, label="prior (all sims)")
    ax[1].hist(qs[nn], bins=20, range=(0.5, 1.5), histtype="step", color="C3", lw=2, density=True, label=f"{a.k} nearest in {a.n_pc} PCs: q={rep[name]['nn_q_median']:.2f}")
    ax[1].axvline(1, color="k", ls=":"); ax[1].legend(fontsize=8); ax[1].set_xlabel("q_halo"); ax[1].set_title("ABC posterior in PC space")
    ax[2].hist(maha_sims, bins=60, color="grey", label="sims"); ax[2].axvline(maha, color="C3", lw=2, label=f"real ({rep[name]['mahalanobis_real_pct']:.1f} pct)")
    ax[2].set_xlabel(f"Mahalanobis distance in {a.n_pc}-PC space"); ax[2].legend(fontsize=8); ax[2].set_title("how typical the real stream is")
fig.tight_layout(); fig.savefig(os.path.join(a.out, f"svd_nearest_streams{'_stats' if a.no_occupancy else ''}.png"), dpi=120)
json.dump(rep, open(os.path.join(a.out, f"svd_nearest_streams{'_stats' if a.no_occupancy else ''}.json"), "w"), indent=1)
