#!/usr/bin/env python
"""B-spline prior-predictive check + nearest neighbours on spline values.

Per stream: simulated rows go through the TRAINING observation model (``--aug``, up to mask_vlos),
the real members through their preset; both are projected into the published STREAMFINDER frame.
Each observable (phi2, parallax, mu_phi1, mu_phi2, v_los[measured stars]) is fitted vs phi1 with a
cubic LSQ B-spline on a FIXED knot vector (interior knots = equal-count phi1 quantiles of the REAL
members, restricted to the real phi1 range) and evaluated on a uniform phi1 grid inside that range,
so every realization yields the same feature vector. PPC = the real spline against the sim spline
band; NN = k nearest rows in robust-standardized grid-value space, their parameters vs the prior.

  .venv/bin/python scripts/ppc_bspline_nn.py --out outputs/Bsline/palau23_dr3/ppc_rnbody_v4
"""
import argparse, json, os, sys
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0"); os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import NAMES, augment_sim  # noqa: E402
from ppc_particle_coverage import real_clouds  # noqa: E402
from ppc_observation_space import load_sim_groups, stream_frames, to_obs  # noqa: E402
from nearest_training_streams import GLOBALS as _G, LOCALS  # noqa: E402
GLOBALS = _G + ["rho_TwoPowerTriaxial_halo", "a_TwoPowerTriaxial_halo", "beta_TwoPowerTriaxial_halo"]  # + legacy halo keys

SPACES = {"stream": ["phi2 [deg]", "parallax [mas]", "mu_phi1 [mas/yr]", "mu_phi2 [mas/yr]", "v_los [km/s]"],
          "icrs": ["ra [deg]", "dec [deg]", "parallax [mas]", "mu_ra* [mas/yr]", "mu_dec [mas/yr]", "v_los [km/s]"]}
K = 3


def table(R, stars, space):
    """(N,6) ICRS catalogue -> (N, 1+n_obs): phi1 (frame abscissa) then the observables of ``space``."""
    F = to_obs(R, stars)                                     # phi1, phi2, plx, mu1, mu2, vlos
    if space == "stream":
        return F
    st = np.asarray(stars, float)
    return np.column_stack([F[:, 0], st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4], st[:, 5]])


def knots(x, n_int):
    q = np.quantile(x, np.linspace(0, 1, n_int + 2))
    return np.r_[[q[0]] * (K + 1), q[1:-1], [q[-1]] * (K + 1)]


def spline_on_grid(x, y, t, grid, min_per_span=2):
    """Fixed-knot cubic LSQ B-spline evaluated on ``grid``; NaN if the data under-populate a span."""
    m = np.isfinite(y) & (x >= t[0]) & (x <= t[-1])
    x, y = x[m], y[m]
    if len(x) < len(t) - K - 1 + 2 or np.any(np.histogram(x, np.unique(t))[0] < min_per_span):
        return np.full(len(grid), np.nan)
    o = np.argsort(x); x, y = x[o], y[o]
    if x[0] > t[0] or x[-1] < t[-1]:  # clamped ends must sit inside the data
        x, y = np.r_[t[0], x, t[-1]], np.r_[y[0], y, y[-1]]
    try:
        return make_lsq_spline(x, y, t, k=K)(grid)
    except (ValueError, np.linalg.LinAlgError):
        return np.full(len(grid), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/data_jarvis/data_agama_rnbody_ibata_m200c_v4_hydrabflow/training_data_100000.npz")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_palau23_dr3.npz")
    ap.add_argument("--simulator", default="stream_agama_rnbody_ibata_m200c_v4")
    ap.add_argument("--aug", default="stream_global_palau23_dr3_emperr")
    ap.add_argument("--real-aug", default="stream_real_global_palau23_dr3_emperr")
    ap.add_argument("--n-sim", type=int, default=2000, help="realizations per stream")
    ap.add_argument("--n-grid", type=int, default=20); ap.add_argument("--stars-per-knot", type=int, default=25)
    ap.add_argument("--k", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--space", default="stream", choices=list(SPACES), help="observables fitted vs phi1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); OBS = {i + 1: lab for i, lab in enumerate(SPACES[a.space])}; ASTRO = tuple(list(OBS)[:-1]); VL = list(OBS)[-1]; os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)

    real = real_clouds(a.real, a.simulator, a.real_aug, 1000, a.seed)
    R_of = stream_frames("streamfinder", a.real)
    sim_raw, jj = load_sim_groups(a.sim, a.n_sim, rng)
    G = sim_raw.shape[0]
    # per-(row) parameter arrays + the flat row index of every (group, stream) that load_sim_groups drew
    with np.load(a.sim) as d:
        j_raw = np.asarray(d["j"]); grouped = d["sim_data_projected"].ndim == 4
        if grouped:                       # globals (n,1), locals (n,m,1) -> flatten to n*m rows
            n, m = j_raw.reshape(j_raw.shape[0], -1).shape
            jf = j_raw.reshape(n, m).astype(int).reshape(-1)
            params = {k: (np.repeat(np.asarray(d[k]).reshape(n, -1)[:, 0], m) if np.asarray(d[k]).reshape(n, -1).shape[1] == 1
                          else np.asarray(d[k]).reshape(n, m, -1)[:, :, 0].reshape(-1)) for k in GLOBALS + LOCALS if k in d.files}
            pick = np.sort(np.random.default_rng(a.seed).choice(n, size=min(a.n_sim, n), replace=False))
            row_of = pick[:, None] * m + np.arange(m)[None]
        else:
            jf = j_raw.reshape(-1).astype(int)
            params = {k: np.asarray(d[k]).reshape(len(jf), -1)[:, 0] for k in GLOBALS + LOCALS if k in d.files}
            rng2 = np.random.default_rng(a.seed)
            per = [rng2.choice(np.flatnonzero(jf == s), size=min(a.n_sim, int((jf == s).sum())), replace=False) for s in range(3)]
            row_of = np.stack([np.sort(p[:G]) for p in per], 1)
    assert np.all(jf[row_of] == jj), "row bookkeeping does not reproduce load_sim_groups' draw"
    sim, attn, vmask = augment_sim(sim_raw, jj, aug_preset=a.aug, simulator=a.simulator, seed=a.seed, upto="mask_vlos")

    rep = dict(sim=a.sim, real=a.real, aug=a.aug, space=a.space, n_sim=G, n_grid=a.n_grid, k=a.k)
    store = {}
    for j, name in NAMES.items():
        Fr, r_vm = table(R_of[j], real[j][0], a.space), real[j][1]
        lo, hi = Fr[:, 0].min(), Fr[:, 0].max()
        grid = np.linspace(lo, hi, a.n_grid)
        t = knots(Fr[:, 0], int(np.clip(len(Fr) // a.stars_per_knot, 1, 6)))
        tv = knots(Fr[r_vm, 0], 1 if r_vm.sum() >= 12 else 0)
        gridv = np.linspace(tv[0], tv[-1], a.n_grid)
        real_f = {c: spline_on_grid(Fr[:, 0], Fr[:, c], t, grid) for c in ASTRO}
        real_f[VL] = spline_on_grid(Fr[r_vm, 0], Fr[r_vm, VL], tv, gridv)
        sim_f = {c: np.full((G, a.n_grid), np.nan) for c in OBS}
        n_att = np.zeros(G, int)
        for g in range(G):
            at = attn[g, j]; n_att[g] = at.sum()
            if at.sum() < 8: continue
            F = table(R_of[j], sim[g, j][at].astype(float), a.space); vm = vmask[g, j][at]
            for c in ASTRO:
                sim_f[c][g] = spline_on_grid(F[:, 0], F[:, c], t, grid)
            sim_f[VL][g] = spline_on_grid(F[vm, 0], F[vm, VL], tv, gridv)

        # ---- PPC: real spline vs sim band -------------------------------------------------
        out = dict(n_real=int(len(Fr)), n_real_vlos=int(r_vm.sum()), phi1_range=[float(lo), float(hi)],
                   interior_knots=int(len(t) - 2 * (K + 1)), usable={}, inside_90={}, real_pct_median={})
        fig, axes = plt.subplots(len(OBS), 1, figsize=(9, 2.8 * len(OBS)), sharex=True)
        for ax, c in zip(axes, OBS):
            S = sim_f[c]; ok = np.all(np.isfinite(S), 1); S = S[ok]; gx = gridv if c == VL else grid
            out["usable"][OBS[c]] = int(ok.sum())
            if ok.sum() >= 20:
                q = np.percentile(S, [5, 16, 50, 84, 95], axis=0)
                ax.fill_between(gx, q[0], q[4], color="C0", alpha=0.2, label="sim 5-95 %")
                ax.fill_between(gx, q[1], q[3], color="C0", alpha=0.35, label="sim 16-84 %")
                ax.plot(gx, q[2], color="C0", lw=1.5, label="sim median")
                pct = np.mean(S <= real_f[c][None], 0) * 100
                out["inside_90"][OBS[c]] = float(np.mean((real_f[c] >= q[0]) & (real_f[c] <= q[4])))
                out["real_pct_median"][OBS[c]] = float(np.median(pct))
                ax.text(0.01, 0.95, f"real inside 5-95 %: {out['inside_90'][OBS[c]]:.2f}; median pct {np.median(pct):.0f}; usable sims {ok.sum()}",
                        transform=ax.transAxes, va="top", fontsize=8)
            sel = r_vm if c == VL else np.ones(len(Fr), bool)
            ax.scatter(Fr[sel, 0], Fr[sel, c], s=8, color="0.3", alpha=0.6, label="real members")
            ax.plot(gx, real_f[c], color="C3", lw=2, label="real B-spline")
            ax.set_ylabel(OBS[c])
            v = Fr[sel, c]; l_, h_ = np.percentile(v, [2, 98]); pad = 0.6 * (h_ - l_) + 1e-3; ax.set_ylim(l_ - pad, h_ + pad)
        axes[0].legend(fontsize=7, loc="lower right"); axes[-1].set_xlabel("phi1 [deg]")
        axes[0].set_title(f"{name}: B-spline PPC, {G} sims through {a.aug} (fixed knots, real phi1 range, {a.space} observables)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"ppc_{name}.png"), dpi=130); plt.close(fig)

        # ---- NN on grid values -----------------------------------------------------------
        feats = {"astrometry": ASTRO, "all": tuple(OBS)}
        dist = {}
        for lab, cs in feats.items():
            X = np.concatenate([sim_f[c] for c in cs], 1); xr = np.concatenate([real_f[c] for c in cs])
            mu = np.nanmedian(X, 0); sc = 1.4826 * np.nanmedian(np.abs(X - mu), 0) + 1e-9
            Z = (X - mu) / sc; zr = (xr - mu) / sc
            fin = np.all(np.isfinite(Z), 1)
            d = np.full(G, np.inf); d[fin] = np.sqrt(np.mean((Z[fin] - zr) ** 2, 1))
            dist[lab] = d
        d = dist["astrometry"]; nn = np.argsort(d)[: a.k]; nn = nn[np.isfinite(d[nn])]
        rows = row_of[:, j]
        # typicality: real->nearest sim vs sim->nearest sim (same feature space)
        Xa = np.concatenate([sim_f[c] for c in feats["astrometry"]], 1); fin = np.all(np.isfinite(Xa), 1)
        mu = np.nanmedian(Xa, 0); sc = 1.4826 * np.nanmedian(np.abs(Xa - mu), 0) + 1e-9; Za = ((Xa - mu) / sc)[fin]
        pick = rng.choice(len(Za), size=min(300, len(Za)), replace=False)
        D = np.sqrt(np.mean((Za[pick][:, None] - Za[None]) ** 2, -1)); D[np.arange(len(pick)), pick] = np.inf
        null_nn = D.min(1)
        out.update(nn_rows=rows[nn].tolist(), nn_dist=d[nn].tolist(), nearest_dist=float(d[nn[0]]),
                   sim_nearest_sim_median=float(np.median(null_nn)), real_nearest_pct=float(100 * np.mean(null_nn <= d[nn[0]])), params={})
        print(f"\n== {name}: usable {out['usable']}; real->nearest sim {d[nn[0]]:.3f} vs sim->nearest sim median {np.median(null_nn):.3f} "
              f"(real at the {out['real_nearest_pct']:.0f}th pct)")
        print(f"  {'param':36s} {'prior med [16,84]':>28s}   {'k-NN med [16,84]':>28s}  shift/prior_sd")
        for k in GLOBALS + LOCALS:
            if k not in params: continue
            pv = params[k][jf == j]; nv = params[k][rows[nn]]
            pv, nv = pv[np.isfinite(pv)], nv[np.isfinite(nv)]
            if not len(nv) or pv.std() == 0: continue
            pq, nq = np.percentile(pv, [50, 16, 84]), np.percentile(nv, [50, 16, 84]); z = (nq[0] - pq[0]) / pv.std()
            out["params"][k] = dict(prior=pq.tolist(), nn=nq.tolist(), shift_sd=float(z))
            print(f"  {k:36s} {pq[0]:9.3g} [{pq[1]:8.3g},{pq[2]:8.3g}]   {nq[0]:9.3g} [{nq[1]:8.3g},{nq[2]:8.3g}]  {z:+.2f}")
        rep[name] = out
        store.update({f"{name}/grid": grid, f"{name}/grid_vlos": gridv, f"{name}/knots": t, f"{name}/nn_rows": rows[nn], f"{name}/nn_dist": d[nn],
                      **{f"{name}/real_{c}": real_f[c] for c in OBS}, **{f"{name}/sim_{c}": sim_f[c] for c in OBS}, f"{name}/rows": rows})

        fig, axes = plt.subplots(len(OBS), 1, figsize=(9, 2.8 * len(OBS)), sharex=True)
        for ax, c in zip(axes, OBS):
            gx = gridv if c == VL else grid
            for r_, g in enumerate(nn[:10]):
                ax.plot(gx, sim_f[c][g], color=f"C{r_ % 10}", lw=1, alpha=0.8, label=f"row {rows[g]} d={d[g]:.2f}" if c == 1 else None)
            sel = r_vm if c == VL else np.ones(len(Fr), bool)
            ax.scatter(Fr[sel, 0], Fr[sel, c], s=8, color="0.3", alpha=0.6); ax.plot(gx, real_f[c], color="k", lw=2.5, label="real" if c == 1 else None)
            ax.set_ylabel(OBS[c]); v = Fr[sel, c]; l_, h_ = np.percentile(v, [2, 98]); pad = 0.6 * (h_ - l_) + 1e-3; ax.set_ylim(l_ - pad, h_ + pad)
        axes[0].legend(fontsize=6, ncol=2); axes[-1].set_xlabel("phi1 [deg]"); axes[0].set_title(f"{name}: 10 nearest rows (astrometry spline distance)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"nearest_{name}.png"), dpi=130); plt.close(fig)

        keys = [k for k in out["params"] if k in GLOBALS]
        if keys:
            fig, ax = plt.subplots(2, 5, figsize=(20, 7)); ax = ax.ravel()
            for i, k in enumerate(keys[:10]):
                pv = params[k][jf == j]; nv = params[k][rows[nn]]
                ax[i].hist(pv[np.isfinite(pv)], bins=30, density=True, color="0.7", label="prior (training set)")
                ax[i].hist(nv[np.isfinite(nv)], bins=15, density=True, histtype="step", color="C3", lw=2, label=f"{len(nn)} nearest")
                ax[i].set_title(k, fontsize=9)
            ax[0].legend(fontsize=7); fig.suptitle(f"{name}: globals of the {len(nn)} nearest rows (B-spline grid NN) vs prior")
            fig.tight_layout(); fig.savefig(os.path.join(a.out, f"nearest_params_{name}.png"), dpi=110); plt.close(fig)

    np.savez(os.path.join(a.out, "bspline_features.npz"), **store)
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
