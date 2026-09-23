#!/usr/bin/env python
"""Nearest training realizations to the real members, in observation space.

Per stream: draw --n-rows rows of a FLAT training set, push them through the training observation
model (``--aug``, up to mask_vlos), the real members through their preset, project both into the
PUBLISHED STREAMFINDER frames, and rank the rows by RBF-MMD^2 to the real cloud on
[phi1, phi2, parallax, mu_phi1, mu_phi2] (all attended stars; v_los reported separately on measured
stars). The k nearest rows' parameters are reported against the prior, and the 6 nearest are
overplotted on the members. Reuses ppc_observation_space / ppc_particle_coverage helpers.

  .venv/bin/python scripts/nearest_training_streams.py --out outputs/nn_v4_rnbody_palau23
"""
import argparse, json, os, sys
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0"); os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import NAMES, augment_sim  # noqa: E402
from ppc_particle_coverage import real_clouds, mmd2, features  # noqa: E402
from ppc_observation_space import read_rows, stream_frames, to_obs  # noqa: E402

OBS = ["phi1", "phi2", "parallax", "mu_phi1", "mu_phi2", "vlos"]
GLOBALS = ["log10_M200_TwoPowerTriaxial_halo", "ln_cvprime_TwoPowerTriaxial_halo", "gamma_TwoPowerTriaxial_halo",
           "q_TwoPowerTriaxial_halo", "p_TwoPowerTriaxial_halo", "alpha_TwoPowerTriaxial_halo",
           "r_Disk", "z_Disk", "Sigma_Disk", "rho_Bulge"]
LOCALS = ["t_end", "m_progenitor", "m_bound_final", "r", "vr"]


def std_feat(c, cols, mu, sc):
    """Standardized selected columns of a (stars, vlos_mask) cloud, or None if too few stars."""
    if c is None:
        return None
    Z = (features(c[0], c[1], cols) - mu) / sc
    return Z if len(Z) >= 10 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/data_jarvis/data_agama_rnbody_ibata_m200c_v4_hydrabflow/training_data_100000.npz")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_palau23_dr3.npz")
    ap.add_argument("--simulator", default="stream_agama_rnbody_ibata_m200c_v4")
    ap.add_argument("--aug", default="stream_global_palau23_dr3_emperr")
    ap.add_argument("--real-aug", default="stream_real_global_palau23_dr3_emperr")
    ap.add_argument("--frame", default="streamfinder", choices=("streamfinder", "palau", "fit"))
    ap.add_argument("--n-rows", type=int, default=3000, help="training rows per stream")
    ap.add_argument("--k", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)

    real = real_clouds(a.real, a.simulator, a.real_aug, 1000, a.seed)
    R_of = stream_frames(a.frame, a.real)

    with np.load(a.train) as d:
        jf = np.asarray(d["j"]).reshape(-1).astype(int)
        params = {k: np.asarray(d[k]).reshape(len(jf), -1)[:, 0] for k in GLOBALS + LOCALS if k in d.files}
    a.n_rows = min(a.n_rows, *[int((jf == j).sum()) for j in range(3)])   # --n-rows 0/huge = the whole set
    rows = [np.sort(rng.choice(np.flatnonzero(jf == j), size=a.n_rows, replace=False)) for j in range(3)]
    flat = read_rows(a.train, "sim_data_projected", np.concatenate(rows))
    sim = np.stack([flat[k * a.n_rows:(k + 1) * a.n_rows] for k in range(3)], axis=1)   # (G,3,P,6)
    jj = np.tile(np.arange(3), (a.n_rows, 1))
    sim, attn, vmask = augment_sim(sim, jj, aug_preset=a.aug, simulator=a.simulator, seed=a.seed, upto="mask_vlos")

    nn_store = {}
    rep = {"train": a.train, "real": a.real, "aug": a.aug, "frame": a.frame, "n_rows": a.n_rows, "k": a.k}
    for j, name in NAMES.items():
        Fr = to_obs(R_of[j], real[j][0]); r_vm = real[j][1]
        clouds = []
        for g in range(a.n_rows):
            at = attn[g, j].astype(bool)
            clouds.append((to_obs(R_of[j], sim[g, j][at].astype(float)), vmask[g, j][at].astype(bool)) if at.sum() >= 10 else None)
        dist = {}
        for sub, cols in {"astrometry": [0, 1, 2, 3, 4], "vlos": [0, 5]}.items():
            Xr = features(Fr, r_vm, cols); mu = np.median(Xr, 0); sc = 1.4826 * np.median(np.abs(Xr - mu), 0) + 1e-9
            Zr = (Xr - mu) / sc; d2 = np.sum((Zr[:, None] - Zr[None]) ** 2, -1); gamma = 1.0 / (2 * np.median(d2[d2 > 0]))
            Zs = [std_feat(c, cols, mu, sc) for c in clouds]
            dist[sub] = np.array([mmd2(Zr, Z, gamma) if Z is not None else np.nan for Z in Zs])
            if sub == "astrometry":
                Zast, gast = Zs, gamma
        d = dist["astrometry"]; order = np.argsort(np.where(np.isfinite(d), d, np.inf)); nn = order[: a.k]
        idx = rows[j]
        # how typical is the real cloud? compare its nearest distance to sims' nearest-to-each-other
        pick = rng.choice(np.flatnonzero(np.isfinite(d)), size=min(60, int(np.isfinite(d).sum())), replace=False)
        null_nn = [np.min([mmd2(Zast[p], Zast[q], gast) for q in pick if q != p]) for p in pick]
        real_nn_pool = float(np.min(d[pick]))  # same-size pool as the null, so the two are comparable
        out = dict(n_usable=int(np.isfinite(d).sum()), nearest_mmd2=float(d[nn[0]]), median_mmd2=float(np.nanmedian(d)),
                   real_nearest_in_pool=real_nn_pool, sim_to_nearest_sim_mmd2_median=float(np.median(null_nn)),
                   real_nearest_pool_pct=float(100 * np.mean(np.array(null_nn) <= real_nn_pool)),
                   nn_rows=idx[nn].tolist(), nn_mmd2_astrometry=d[nn].tolist(), nn_mmd2_vlos=dist["vlos"][nn].tolist(), params={})
        print(f"\n== {name}: {out['n_usable']} usable rows; nearest MMD^2 {d[nn[0]]:.4f}, median {np.nanmedian(d):.4f}; "
              f"in a 60-row pool: real->nearest sim {real_nn_pool:.4f} vs sim->nearest sim median {np.median(null_nn):.4f} (real at the {out['real_nearest_pool_pct']:.0f}th pct)")
        print(f"  {'param':36s} {'prior med [16,84]':>28s}   {'k-NN med [16,84]':>28s}  shift/prior_sd")
        for k in GLOBALS + LOCALS:
            if k not in params: continue
            pv = params[k][jf == j]; nv = params[k][idx[nn]]
            pv, nv = pv[np.isfinite(pv)], nv[np.isfinite(nv)]
            if not len(nv) or pv.std() == 0: continue
            q = lambda v: np.percentile(v, [50, 16, 84])
            pq, nq = q(pv), q(nv); z = (nq[0] - pq[0]) / pv.std()
            out["params"][k] = dict(prior=pq.tolist(), nn=nq.tolist(), shift_sd=float(z), frac_nonzero=float(np.mean(nv > 0)))
            print(f"  {k:36s} {pq[0]:9.3g} [{pq[1]:8.3g},{pq[2]:8.3g}]   {nq[0]:9.3g} [{nq[1]:8.3g},{nq[2]:8.3g}]  {z:+.2f}")
        rep[name] = out
        nn_store[j] = (Fr, r_vm, [clouds[g] for g in nn], d[nn])
        np.savez(os.path.join(a.out, f"nn_clouds_{name}.npz"), rows=idx[nn], mmd2=d[nn],
                 stars=np.concatenate([clouds[g][0] for g in nn]), vlos_mask=np.concatenate([clouds[g][1] for g in nn]),
                 rank=np.concatenate([np.full(len(clouds[g][0]), r) for r, g in enumerate(nn)]), real=Fr, real_vlos_mask=r_vm)

        fig, ax = plt.subplots(1, 5, figsize=(22, 4.2))
        for c in range(1, 6):
            for rank, g in enumerate(nn[:6]):
                F, vm = clouds[g]; s = vm if c == 5 else np.ones(len(F), bool)
                ax[c - 1].scatter(F[s, 0], F[s, c], s=4, alpha=0.5, color=f"C{rank}", label=f"row {idx[g]} (mmd² {d[g]:.3f})" if c == 1 else None)
            s = r_vm if c == 5 else np.ones(len(Fr), bool)
            ax[c - 1].scatter(Fr[s, 0], Fr[s, c], s=14, color="k", marker="x", label="real" if c == 1 else None)
            ax[c - 1].set_xlabel("phi1 [deg]"); ax[c - 1].set_ylabel(OBS[c])
            v = Fr[s, c][np.isfinite(Fr[s, c])]; lo, hi = np.percentile(v, [1, 99]); pad = 0.5 * (hi - lo) + 1e-3
            ax[c - 1].set_ylim(lo - pad, hi + pad)
        ax[0].legend(fontsize=7); fig.suptitle(f"{name}: 6 nearest training rows ({a.frame} frame, {a.aug})")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"nearest_{name}.png"), dpi=120); plt.close(fig)

        keys = [k for k in out["params"] if k in GLOBALS]
        fig, ax = plt.subplots(2, 5, figsize=(20, 7)); ax = ax.ravel()
        for i, k in enumerate(keys[:10]):
            pv = params[k][jf == j]; nv = params[k][idx[nn]]
            ax[i].hist(pv[np.isfinite(pv)], bins=30, density=True, color="0.7", label="prior (training set)")
            ax[i].hist(nv[np.isfinite(nv)], bins=15, density=True, histtype="step", color="C3", lw=2, label=f"{a.k} nearest")
            ax[i].set_title(k, fontsize=9)
        ax[0].legend(fontsize=7); fig.suptitle(f"{name}: globals of the {a.k} nearest training rows vs prior")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"nearest_params_{name}.png"), dpi=110); plt.close(fig)
    json.dump(rep, open(os.path.join(a.out, "nearest_training_streams.json"), "w"), indent=1)

    # all k nearest of every stream, pooled, under the real members
    fig, ax = plt.subplots(3, 5, figsize=(22, 11))
    for j, name in NAMES.items():
        Fr, r_vm, cl, dnn = nn_store[j]
        S = np.concatenate([c[0] for c in cl]); VM = np.concatenate([c[1] for c in cl])
        for c in range(1, 6):
            s_ = VM if c == 5 else np.ones(len(S), bool)
            ax[j, c - 1].scatter(S[s_, 0], S[s_, c], s=1, alpha=min(1.0, 40.0 / len(cl)), color="C0", rasterized=True)
            sr = r_vm if c == 5 else np.ones(len(Fr), bool)
            ax[j, c - 1].scatter(Fr[sr, 0], Fr[sr, c], s=12, color="k", marker="x")
            v = Fr[sr, c][np.isfinite(Fr[sr, c])]; lo, hi = np.percentile(v, [1, 99]); pad = 0.5 * (hi - lo) + 1e-3
            ax[j, c - 1].set_ylim(lo - pad, hi + pad); ax[j, c - 1].set_ylabel(OBS[c])
            if j == 2: ax[j, c - 1].set_xlabel("phi1 [deg]")
        ax[j, 0].set_title(f"{name}: {len(cl)} nearest training rows (blue) + real (x); mmd² {dnn[0]:.3f}-{dnn[-1]:.3f}", fontsize=9, loc="left")
    fig.suptitle(f"{a.frame} frame, {a.aug}"); fig.tight_layout(); fig.savefig(os.path.join(a.out, "nearest_all.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
