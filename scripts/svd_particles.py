#!/usr/bin/env python
"""SVD of the PARTICLE representation: noise-free training streams -> PCs; project the noisy test set
(training observation model) and the real Gaia members into the same space.

Per stream, every realization becomes one fixed-length vector: normalized 2-D histograms of
(phi1 x {phi2, parallax, mu_phi1, mu_phi2, v_los}) in the REAL-fitted great-circle frame, on bin edges
set by the real members (v_los from measured stars only). The SVD is fit on NOISE-FREE training rows
(all in-window stars, 1/d parallax, no errors, no subsample); the 333-group test set goes through the
training augmentation chain up to mask_vlos (window, member-count subsample, Gaia errors, v_los
selection, M68 width cut) and the real members through their preset, then both are projected.
Outputs: svd_particles_<stream>.png, svd_particles.json in --out.
"""
import argparse, json, os, sys
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0"); os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import WINDOW, augment_sim, fit_frame  # noqa: E402
from ppc_particle_coverage import real_clouds, to_frame  # noqa: E402

NAMES = ["Pal5", "NGC3201", "M68"]; OBS = ["phi2", "parallax", "mu_phi1", "mu_phi2", "vlos"]
Q = "q_TwoPowerTriaxial_halo"


def edges_from_real(F, vm, k1, k2):
    """phi1 edges from the real member span (+5 %), value edges from robust real percentiles (x1.6)."""
    lo, hi = F[:, 0].min(), F[:, 0].max(); pad = 0.05 * (hi - lo)
    e1 = np.linspace(lo - pad, hi + pad, k1 + 1); e2 = []
    for c in range(1, 6):
        v = F[vm, c] if c == 5 else F[:, c]
        p5, p50, p95 = np.percentile(v, [5, 50, 95]); h = 1.6 * max(p95 - p50, p50 - p5)
        e2.append(np.linspace(p50 - h, p50 + h, k2 + 1))
    return e1, e2


def featurize(F, vm, e1, e2):
    """(N,6) stream-frame stars -> flat vector of 5 normalized phi1 x value histograms."""
    out = []
    for c in range(1, 6):
        sel = vm if c == 5 else np.ones(len(F), bool)
        h, _, _ = np.histogram2d(F[sel, 0], F[sel, c], bins=[e1, e2[c - 1]])
        out.append(h.ravel() / max(h.sum(), 1.0))
    return np.concatenate(out)


def maha(P, ref):
    mu, sd = ref.mean(0), ref.std(0); sd[sd == 0] = 1
    return np.sqrt((((P - mu) / sd) ** 2).sum(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_hydrabflow/training_data_100000.npz")
    ap.add_argument("--test", default="data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_hydrabflow/test_multistream_333.npz")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz")
    ap.add_argument("--simulator", default="stream_agama_spray_massloss_ibata_m200c_v4")
    ap.add_argument("--aug", default="stream_global_v5"); ap.add_argument("--real-aug", default="stream_real_global_v5")
    ap.add_argument("--n-train", type=int, default=4000, help="noise-free training rows per stream")
    ap.add_argument("--k1", type=int, default=16); ap.add_argument("--k2", type=int, default=12)
    ap.add_argument("--n-pc", type=int, default=10); ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="outputs/v5_2modal/q_confusion")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)

    real = real_clouds(a.real, a.simulator, a.real_aug, 300, a.seed)          # {j: (stars, vlos_mask)}
    frames = {j: fit_frame(real[j][0][:, 0], real[j][0][:, 1]) for j in real}
    realF = {j: (to_frame(frames[j], real[j][0]), real[j][1]) for j in real}
    edges = {j: edges_from_real(*realF[j], a.k1, a.k2) for j in real}

    # --- noisy test set through the training chain
    td = np.load(a.test); x = td["sim_data_projected"]; tj = td["j"][..., 0].astype(int); tq = td[Q][:, 0]
    sim, attn, vmask = augment_sim(x, tj, aug_preset=a.aug, simulator=a.simulator, seed=a.seed, upto="mask_vlos")

    # --- noise-free training rows (mmap; keep in-window stars only, parallax = 1/d)
    # npz members are not mmappable: one load of the first 3*n_train rows (~1/3 per stream) instead
    # of re-reading the 4.8 GB array per row
    fl = np.load(a.train); n_head = 3 * a.n_train + 600
    fj = fl["j"][:n_head, 0].astype(int); fq = fl[Q][:n_head, 0]; fx = fl["sim_data_projected"][:n_head]
    rep = {}
    for j, name in enumerate(NAMES):
        e1, e2 = edges[j]; R = frames[j]
        idx = np.where((fj == j) & np.isfinite(fq))[0][: a.n_train]
        Xtr, qtr = [], []
        for i in idx:
            s = np.asarray(fx[i], float); lo_ra, hi_ra, lo_de, hi_de = WINDOW[j]
            ok = (s[:, 0] >= lo_ra) & (s[:, 0] <= hi_ra) & (s[:, 1] >= lo_de) & (s[:, 1] <= hi_de) & np.isfinite(s).all(1)
            if ok.sum() < 20: continue
            s = s[ok].copy(); s[:, 2] = 1.0 / s[:, 2]
            Xtr.append(featurize(to_frame(R, s), np.ones(len(s), bool), e1, e2)); qtr.append(fq[i])
        Xtr, qtr = np.array(Xtr), np.array(qtr)
        Xte, qte = [], []
        for g in range(sim.shape[0]):
            m = np.where(tj[g] == j)[0]
            if not len(m): continue
            st, at, vm = sim[g, m[0]], attn[g, m[0]], vmask[g, m[0]]
            if at.sum() < 20 or not np.isfinite(tq[g]): continue
            Xte.append(featurize(to_frame(R, st[at].astype(float)), vm[at], e1, e2)); qte.append(tq[g])
        Xte, qte = np.array(Xte), np.array(qte)
        xr = featurize(*realF[j], e1, e2)

        mu = Xtr.mean(0); U, sv, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[: a.n_pc].T
        var = sv ** 2 / (sv ** 2).sum()
        Ptr, Pte, pr = (Xtr - mu) @ V, (Xte - mu) @ V, (xr - mu) @ V
        # where does the real point land, relative to the noise-free cloud and to the noisy test cloud?
        m_tr = maha(Ptr, Ptr); m_te_vs_tr = maha(Pte, Ptr); m_r_vs_tr = float(maha(pr[None], Ptr)[0])
        m_te = maha(Pte, Pte); m_r_vs_te = float(maha(pr[None], Pte)[0])
        pct_tr = [100 * np.mean(Ptr[:, i] <= pr[i]) for i in range(a.n_pc)]
        pct_te = [100 * np.mean(Pte[:, i] <= pr[i]) for i in range(a.n_pc)]
        # noise shift: does the observation model move the test set off the noise-free cloud?
        d_tr = np.sqrt(((Ptr - pr) ** 2).sum(1)); nn_tr = np.argsort(d_tr)[: a.k]
        d_te = np.sqrt(((Pte - pr) ** 2).sum(1)); nn_te = np.argsort(d_te)[: max(10, a.k // 10)]
        rho = [float(np.corrcoef(Ptr[:, i], qtr)[0, 1]) for i in range(a.n_pc)]
        A = np.c_[Pte, np.ones(len(qte))]; coef, *_ = np.linalg.lstsq(A, qte, rcond=None)
        r2 = 1 - np.mean((A @ coef - qte) ** 2) / qte.var(); q_lin = float(np.r_[pr, 1] @ coef)
        blocks = a.k1 * a.k2
        loads = [{OBS[b]: float((V[b * blocks:(b + 1) * blocks, i] ** 2).sum()) for b in range(5)} for i in range(a.n_pc)]
        rep[name] = dict(n_train=len(qtr), n_test=len(qte), var_first_n=float(var[: a.n_pc].sum()),
                         real_maha_vs_noisefree=m_r_vs_tr, real_maha_pct_vs_noisefree=float(100 * np.mean(m_tr <= m_r_vs_tr)),
                         noisy_test_maha_vs_noisefree_median=float(np.median(m_te_vs_tr)),
                         real_maha_vs_noisy_test=m_r_vs_te, real_maha_pct_vs_noisy_test=float(100 * np.mean(m_te <= m_r_vs_te)),
                         pc_pct_real_vs_noisefree=pct_tr, pc_pct_real_vs_noisy_test=pct_te, pc_corr_q=rho, pc_loads=loads,
                         nn_noisefree_q=dict(median=float(np.median(qtr[nn_tr])), p16=float(np.percentile(qtr[nn_tr], 16)), p84=float(np.percentile(qtr[nn_tr], 84)), frac_prolate=float(np.mean(qtr[nn_tr] > 1))),
                         nn_noisy_test_q=dict(median=float(np.median(qte[nn_te])), p16=float(np.percentile(qte[nn_te], 16)), p84=float(np.percentile(qte[nn_te], 84)), frac_prolate=float(np.mean(qte[nn_te] > 1)), k=len(nn_te)),
                         linear_q_readout_r2_noisy_test=float(r2), linear_q_readout_real=q_lin)
        print(f"\n== {name}: {len(qtr)} noise-free training rows, {len(qte)} noisy test rows; first {a.n_pc} PCs = {100*var[:a.n_pc].sum():.0f}% of variance")
        for i in range(a.n_pc):
            top = sorted(loads[i].items(), key=lambda kv: -kv[1])[:2]
            print(f"  PC{i+1:2d} {100*var[i]:4.1f}%  corr(q) {rho[i]:+.2f}  real pct vs noise-free {pct_tr[i]:5.1f} / vs noisy test {pct_te[i]:5.1f}   loads " + ", ".join(f"{k} {v:.2f}" for k, v in top))
        print(f"  noisy test vs noise-free cloud: median Mahalanobis {np.median(m_te_vs_tr):.1f} (noise-free rows themselves {np.median(m_tr):.1f})")
        print(f"  REAL vs noise-free cloud: Mahalanobis {m_r_vs_tr:.1f} ({rep[name]['real_maha_pct_vs_noisefree']:.1f} pct);  vs noisy test cloud: {m_r_vs_te:.1f} ({rep[name]['real_maha_pct_vs_noisy_test']:.1f} pct)")
        print(f"  q of {a.k} nearest noise-free rows {rep[name]['nn_noisefree_q']['median']:.2f} [{rep[name]['nn_noisefree_q']['p16']:.2f},{rep[name]['nn_noisefree_q']['p84']:.2f}] P(q>1) {rep[name]['nn_noisefree_q']['frac_prolate']:.2f};  "
              f"of {len(nn_te)} nearest noisy test groups {rep[name]['nn_noisy_test_q']['median']:.2f} [{rep[name]['nn_noisy_test_q']['p16']:.2f},{rep[name]['nn_noisy_test_q']['p84']:.2f}] P(q>1) {rep[name]['nn_noisy_test_q']['frac_prolate']:.2f}")
        print(f"  linear q readout fitted on the noisy test PCs: R^2 {r2:.2f}; real -> q = {q_lin:.2f}")

        fig, ax = plt.subplots(2, 3, figsize=(19, 11))
        for k, (i, jj) in enumerate([(0, 1), (2, 3), (4, 5)]):
            if jj >= a.n_pc: break
            ax[0, k].scatter(Ptr[:, i], Ptr[:, jj], s=3, c="0.6", label="noise-free training")
            sc = ax[0, k].scatter(Pte[:, i], Pte[:, jj], s=14, c=qte, cmap="coolwarm", vmin=0.5, vmax=1.5, edgecolor="k", linewidth=0.2, label="noisy test (colour = q)")
            ax[0, k].plot(pr[i], pr[jj], "*", color="lime", ms=20, mec="k", label="real Gaia")
            ax[0, k].set_xlabel(f"PC{i+1}"); ax[0, k].set_ylabel(f"PC{jj+1}"); ax[0, k].legend(fontsize=7, loc="best")
        fig.colorbar(sc, ax=ax[0, :], fraction=0.015, label="true q (test)")
        ax[1, 0].hist(m_tr, bins=50, density=True, color="0.6", label="noise-free training"); ax[1, 0].hist(m_te_vs_tr, bins=50, density=True, histtype="step", color="C0", lw=2, label="noisy test")
        ax[1, 0].axvline(m_r_vs_tr, color="lime", lw=3, label=f"real ({rep[name]['real_maha_pct_vs_noisefree']:.0f} pct of noise-free)"); ax[1, 0].set_xlabel(f"Mahalanobis to the noise-free cloud ({a.n_pc} PCs)"); ax[1, 0].legend(fontsize=8)
        ax[1, 1].hist(m_te, bins=40, density=True, color="C0", alpha=0.6, label="noisy test"); ax[1, 1].axvline(m_r_vs_te, color="lime", lw=3, label=f"real ({rep[name]['real_maha_pct_vs_noisy_test']:.0f} pct)")
        ax[1, 1].set_xlabel(f"Mahalanobis to the noisy TEST cloud ({a.n_pc} PCs)"); ax[1, 1].legend(fontsize=8)
        ax[1, 2].hist(qtr, bins=20, range=(0.5, 1.5), density=True, histtype="step", color="0.5", label="prior")
        ax[1, 2].hist(qtr[nn_tr], bins=20, range=(0.5, 1.5), density=True, histtype="step", color="C3", lw=2, label=f"{a.k} nearest noise-free: q={np.median(qtr[nn_tr]):.2f}")
        ax[1, 2].hist(qte[nn_te], bins=10, range=(0.5, 1.5), density=True, histtype="step", color="C0", lw=2, label=f"{len(nn_te)} nearest noisy test: q={np.median(qte[nn_te]):.2f}")
        ax[1, 2].axvline(1, color="k", ls=":"); ax[1, 2].set_xlabel("q_halo"); ax[1, 2].legend(fontsize=8)
        fig.suptitle(f"{name}: SVD of noise-free particle histograms (phi1 x [phi2, plx, mu_phi1, mu_phi2, vlos], {a.k1}x{a.k2} bins, real-fitted frame); "
                     f"first {a.n_pc} PCs = {100*var[:a.n_pc].sum():.0f}% var", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"svd_particles_{name}.png"), dpi=120); plt.close(fig)
    json.dump(rep, open(os.path.join(a.out, "svd_particles.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
