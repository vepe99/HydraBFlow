#!/usr/bin/env python
"""Prior-predictive check with one fit PER OBSERVABLE per stream realization, each on its own support.

phi2, parallax, mu_phi1, mu_phi2: cubic LSQ B-spline vs phi1 with ``--n-interior`` interior knots at the
equal-count phi1 quantiles of THAT realization's stars; v_los: 2nd-order polynomial over its measured
stars. Every curve lives only on [min, max] phi1 of the stars it was fitted to (no clamping to the real
range, no window cut). Sims = stored training rows (RA/Dec-windowed at storage time, cannot be undone),
de-padded, 1/d, randomly subsampled to the REAL member count (v_los kept for the real measured count),
Gaia DR3 noise from the training steps (``--no-noise`` to skip); STREAMFINDER frames. The figure draws
every sim curve on its own support; the report scores the real curve on 20 of the real phi1 values,
only where >= ``--min-sim-cov`` of the sims reach.

  .venv/bin/python scripts/ppc_stream_splines.py --out outputs/stream_splines/prog2026_ou24vc_noise
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_joint_bspline import NAMES, LABELS, NOISE_STEPS, gaia_noise, knots, sf, to_frame  # noqa: E402  (sets the CPU env)
import numpy as np, matplotlib  # noqa: E402
matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.interpolate import make_lsq_spline, make_smoothing_spline  # noqa: E402
from ppc_observation_space import read_rows  # noqa: E402

VL = 4
def smooth(x, y, lam):
    """Penalized cubic smoothing spline (sum (y-f)^2 + lam * int f''^2); near-duplicate phi1 merged with multiplicity weights."""
    xu, inv = np.unique(np.round(x, 4), return_inverse=True)
    cnt = np.bincount(inv); yu = np.bincount(inv, y) / cnt
    if len(xu) < 5:
        raise ValueError
    return make_smoothing_spline(xu, yu, w=cnt.astype(float), lam=lam)


def norm_lams(F, s_track, s_plx):
    """{c: lambda} for the astrometric columns, lambda = s * N * L^3 with N, L = star count and phi1 length of the REAL
    members, so s is dimensionless and comparable across streams; the same lambda is reused for bootstrap and every sim."""
    scale = len(F) * np.ptp(F[:, 0]) ** 3
    return {c: (s_plx if c == 1 else s_track) * scale for c in range(VL)}


def fit_curves(F, vmask, n_int, vlos_deg=2, lams=None, plx_deg=2, core_frac=None):
    """(N,6) frame table -> 5 callables phi1 -> value (NaN outside the curve's own support), or None if a fit fails."""
    out = []
    core = np.ones(len(F), bool)
    if core_frac is not None and 0 < core_frac < 1:   # this stream sample's central phi1 range holding core_frac of its stars (drops the sparse edges)
        c0, c1 = np.quantile(F[:, 0], [(1 - core_frac) / 2, (1 + core_frac) / 2]); core = (F[:, 0] >= c0) & (F[:, 0] <= c1)
    for c in range(5):
        m = (vmask if c == VL else np.ones(len(F), bool)) & core
        x, y = F[m, 0], F[m, c + 1]
        o = np.argsort(x); x, y = x[o], y[o]
        lo, hi = (x[0], x[-1]) if len(x) else (np.nan, np.nan)
        try:
            deg = {VL: vlos_deg, 1: plx_deg}.get(c)
            if deg is not None and deg >= 0:   # v_los and parallax: low-order polynomial (plx_deg < 0 -> spline like the tracks)
                if len(x) < deg + 2:
                    raise ValueError
                f = np.polynomial.Polynomial.fit(x, y, deg)
            elif lams is not None:
                f = smooth(x, y, lams[c])
            else:
                f = make_lsq_spline(x, y, knots(x, n_int), k=3)
        except (ValueError, np.linalg.LinAlgError):
            out.append(None); continue
        out.append((lambda g, f=f, lo=lo, hi=hi: np.where((g >= lo) & (g <= hi), f(np.clip(g, lo, hi)), np.nan), lo, hi))
    return out


def on_grid(curves, grid):
    return np.stack([np.full(len(grid), np.nan) if cv is None else cv[0](grid) for cv in curves], 1)   # (G,5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_ou24vc_hydrabflow/training_data_300000.npz")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--noise", action=argparse.BooleanOptionalAction, default=True, help="Gaia DR3 errors on the sims (training steps %s)" % (NOISE_STEPS,))
    ap.add_argument("--simulator", default="stream_agama_spray_massloss_ibata_m200c_v4"); ap.add_argument("--noise-preset", default="stream_global")
    ap.add_argument("--n-interior", type=int, default=2, help="interior knots of the astrometric splines")
    ap.add_argument("--fit", default="smoothing", choices=("smoothing", "lsq"), help="astrometric fit: penalized smoothing spline or LSQ B-spline (--n-interior)")
    ap.add_argument("--s-track", type=float, default=1e-4, help="smoothing: dimensionless s for phi2/mu_phi1/mu_phi2, lambda = s * N_real * L_real^3")
    ap.add_argument("--s-parallax", type=float, default=1e-2, help="smoothing: s for parallax (noise dominated -> much stiffer)")
    ap.add_argument("--core-frac", type=float, default=0.85, help="fit (and draw) each curve only on its stream sample's central phi1 range holding this fraction of the stars (<=0 or >=1: all stars)")
    ap.add_argument("--plx-deg", type=int, default=1, help="parallax polynomial degree (-1 = smoothing spline with --s-parallax)")
    ap.add_argument("--vlos-deg", type=int, default=2, help="v_los polynomial degree")
    ap.add_argument("--min-sim-cov", type=float, default=0.5, help="score only grid points reached by >= this fraction of the sims")
    ap.add_argument("--n-sim", type=int, default=1000); ap.add_argument("--n-grid", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-boot", type=int, default=300); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
    R_of = sf.frames()
    noise = gaia_noise(a.simulator, a.noise_preset, a.seed) if a.noise else None

    d = np.load(a.real)
    real_st, att, vm = d["sim_data_projected"][0], d["attention_mask"][:, 0] > 0, d["vlos_mask"] > 0
    j_all = np.asarray(np.load(a.sim)["j"]).reshape(-1).astype(int)
    rep, res = {}, {}
    for j, name in NAMES.items():
        R = R_of[name]
        Fr = to_frame(R, real_st[j][att[j]]); rv = vm[j][att[j]]
        N, Nv = len(Fr), int(rv.sum())
        grid = np.sort(Fr[:, 0])[np.round(np.linspace(0, N - 1, a.n_grid)).astype(int)]   # 20 of the real phi1 values
        lams = norm_lams(Fr, a.s_track, a.s_parallax) if a.fit == "smoothing" else None
        fit = lambda F, v: fit_curves(F, v, a.n_interior, a.vlos_deg, lams, a.plx_deg, a.core_frac)  # noqa: E731
        real_c = fit(Fr, rv); real_f = on_grid(real_c, grid)
        bi = rng.integers(0, N, (a.n_boot, N))   # real-curve uncertainty: star bootstrap (Gaia noise is in the members)
        sig_real = np.nanstd(np.stack([on_grid(fit(Fr[i], rv[i]), grid) for i in bi]), 0)

        # draw rows of this stream until n_sim of them hold >= N stored stars
        cand = rng.permutation(np.flatnonzero(j_all == j)); ST, VM, tried = [], [], 0
        for chunk in np.array_split(cand, len(cand) // 500):
            rows = read_rows(a.sim, "sim_data_projected", np.sort(chunk)); tried += len(chunk)
            for st in rows:
                st = st[np.all(np.isfinite(st), 1) & (st[:, 0] > -900)].astype(float)
                if len(st) < N:
                    continue
                st = st[rng.choice(len(st), N, replace=False)]; st[:, 2] = 1.0 / st[:, 2]   # distance -> parallax
                v = np.zeros(N, bool); v[rng.choice(N, Nv, replace=False)] = True
                ST.append(st); VM.append(v)
                if len(ST) == a.n_sim: break
            if len(ST) == a.n_sim: break
        ST = np.array(ST)
        if noise is not None:
            ST = noise(ST, j)
        sim_c = [fit(to_frame(R, st), v) for st, v in zip(ST, VM)]
        sim_f = np.stack([on_grid(cv, grid) for cv in sim_c])                      # (n_sim, G, 5)

        cov = np.isfinite(sim_f).mean(0)
        keep = (cov >= a.min_sim_cov) & np.isfinite(real_f)                        # scored grid points
        s = np.where(keep[None], sim_f, np.nan)
        fin = np.isfinite(s)
        pct = np.sum(fin & (s < real_f[None]), 0) / np.maximum(fin.sum(0), 1) * 100
        lo, hi = np.nanpercentile(s, 5, 0), np.nanpercentile(s, 95, 0)
        med = np.nanmedian(s, 0); sig = 1.4826 * np.nanmedian(np.abs(s - med), 0)
        z = np.where(keep, (real_f - med) / np.hypot(sig, sig_real), np.nan)
        rep[name] = dict(lam={LABELS[c].split()[0]: v for c, v in (lams or {}).items()}, n_real=N, n_vlos=Nv, rows_tried=tried, rows_used=len(ST), frac_rows_with_enough_stars=len(ST) / tried,
                         **{LABELS[c].split()[0]: dict(n_scored=int(keep[:, c].sum()), fit_failed=float(np.mean([cv[c] is None for cv in sim_c])),
                                                       inside_5_95=float(np.mean(((real_f[:, c] >= lo[:, c]) & (real_f[:, c] <= hi[:, c]))[keep[:, c]])),
                                                       median_pct=float(np.median(pct[keep[:, c], c])), median_z=float(np.nanmedian(z[:, c])),
                                                       max_abs_z=float(np.nanmax(np.abs(z[:, c]))), sim_cov=cov[:, c].round(3).tolist())
                            for c in range(5)})
        res[name] = (Fr, rv, real_c, sim_c)

    fig, ax = plt.subplots(5, 3, figsize=(15, 17), squeeze=False)
    for col, (name, (Fr, rv, real_c, sim_c)) in enumerate(res.items()):
        for c in range(5):
            A = ax[c, col]; ok = rv if c == VL else slice(None)
            segs = [np.c_[g, cv[c][0](g)] for cv in sim_c if cv[c] is not None for g in [np.linspace(cv[c][1], cv[c][2], 60)]]
            A.add_collection(LineCollection(segs, colors="C0", linewidths=0.5, alpha=max(0.02, 20 / len(segs)), label=f"sim fits on own support ({len(segs)})"))
            A.scatter(Fr[ok, 0], Fr[ok, c + 1], s=6, c="k", zorder=3, label="real members")
            if real_c[c] is not None:
                g = np.linspace(real_c[c][1], real_c[c][2], 200)
                A.plot(g, real_c[c][0](g), c="C3", lw=2, zorder=4, label="real fit")
            ys = np.concatenate([sg[:, 1] for sg in segs] + [Fr[ok, c + 1]])
            xs = np.concatenate([sg[[0, -1], 0] for sg in segs] + [Fr[:, 0]])
            A.set_ylim(*np.nanpercentile(ys, [0.5, 99.5])); A.set_xlim(*np.nanpercentile(xs, [1, 99]))
            r = rep[name][LABELS[c].split()[0]]
            A.set_title(f"{name} {LABELS[c].split()[0]}: inside {r['inside_5_95']:.2f} ({r['n_scored']} pts), z_med {r['median_z']:+.2f}", fontsize=9)
            if col == 0: A.set_ylabel(LABELS[c])
            if c == VL: A.set_xlabel("phi1 [deg] (STREAMFINDER frame)")
    ax[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Prior predictive: " + (f"cubic smoothing splines (s = {a.s_track:g}, parallax {a.s_parallax:g})" if a.fit == "smoothing" else f"cubic LSQ B-splines ({a.n_interior} interior knots, own phi1 quantiles)") + f" + parallax poly deg {a.plx_deg} + v_los poly deg {a.vlos_deg}, each on its own support" + (f" (central {a.core_frac:.0%} of phi1)" if 0 < a.core_frac < 1 else "") + "; "
                 + ("Gaia DR3 noise, no window/count cut" if a.noise else "no observation model"), fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "ppc_stream_splines.png"), dpi=120)
    rep["_config"] = vars(a); json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
