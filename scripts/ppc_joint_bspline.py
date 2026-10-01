#!/usr/bin/env python
"""Prior-predictive check with ONE cubic LSQ B-spline for all observables per stream realization.

The ``ppc_bspline_nn.py --fit lsq`` method (cubic B-spline, FIXED knot vector: interior knots at the
equal-count phi1 quantiles of the REAL members, clamped at the real phi1 range, stars outside it
dropped, a row usable only if every knot span holds >= 2 stars), except that one spline maps phi1 ->
(phi2, parallax, mu_phi1, mu_phi2, v_los) jointly: a single design matrix / coefficient array
(n_coef, 5), with v_los rows weighted by the measured-v_los mask (its column needs >= 1 measured star
per span, else that column is NaN for the row).

Sims = stored training rows (already RA/Dec-windowed at storage time), de-padded, 1/d, randomly
subsampled to the REAL member count (v_los kept for the real measured count), optional Gaia DR3 noise
from the training steps (no window/count cut); STREAMFINDER frames. PPC on 20 of the real phi1 values.

  .venv/bin/python scripts/ppc_joint_bspline.py --out outputs/joint_bspline/prog2026_ou24vc_noise
"""
import argparse, importlib.util, json, os, sys
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["HYDRABFLOW_NUM_GPUS"] = "0"; os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_observation_space import read_rows  # noqa: E402  (numpy-only)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("stream_frame", os.path.join(ROOT, "src/hydrabflow/simulators/stream_frame.py"))
sf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sf)   # no hydrabflow/__init__ (would pick a GPU)
NAMES = {0: "Pal5", 1: "NGC3201", 2: "M68"}
LABELS = ["phi2 [deg]", "parallax [mas]", "mu_phi1 [mas/yr]", "mu_phi2 [mas/yr]", "v_los [km/s]"]
K = 3
MAX_COND = 1e4   # design-matrix condition number above which a column's fit is rejected
NOISE_STEPS = ("sample_magnitudes", "sample_obs_error", "apply_obs_error")   # training Gaia error model, no window/count cuts


def gaia_noise(simulator, preset, seed):
    """(n,N,6) ICRS stars with parallax, stream id -> same with Gaia DR3 noise, via the training augmentation steps."""
    from omegaconf import OmegaConf
    from hydrabflow.registry import AUGMENTATIONS
    from ppc_particle_coverage import compose_aug
    params = OmegaConf.create(OmegaConf.to_container(compose_aug(simulator, preset).params, resolve=True))
    steps = [AUGMENTATIONS.get(n)(params, np.random.default_rng(seed)) for n in NOISE_STEPS]

    def apply(st, j):
        b = {"sim_data_projected": np.asarray(st, np.float32), "j": np.full((len(st), 1), j, np.float32)}
        for f in steps:
            b = f(b)
        return np.asarray(b["sim_data_projected"], float)
    return apply


def to_frame(R, st):
    """(N,6) ra, dec, parallax, pmra, pmdec, vlos -> (N,6) phi1, phi2, parallax, mu_phi1, mu_phi2, vlos."""
    p1, p2, m1, m2 = sf.project(R, st[:, 0], st[:, 1], st[:, 3], st[:, 4])
    return np.column_stack([p1, p2, st[:, 2], m1, m2, st[:, 5]])


def knots(x, n_int):
    """Clamped cubic knot vector, interior knots at the equal-count quantiles of x (== ppc_bspline_nn.knots)."""
    q = np.quantile(x, np.linspace(0, 1, n_int + 2))
    return np.r_[[q[0]] * (K + 1), q[1:-1], [q[-1]] * (K + 1)]


def fit_eval(x, Y, W, grid, t, min_per_span=2):
    """x (N,), Y (N,5), W (N,5) 0/1 -> one B-spline phi1 -> R^5 on grid (G,5); NaN column where a span is under-populated."""
    out = np.full((len(grid), Y.shape[1]), np.nan)
    m = (x >= t[0]) & (x <= t[-1]); x, Y, W = x[m], Y[m], W[m]
    edges = np.unique(t)
    if len(x) < len(t) - K - 1 or np.any(np.histogram(x, edges)[0] < min_per_span):
        return out
    B = BSpline.design_matrix(x, t, K).toarray()                                   # (N, n_coef), shared by all columns
    C = np.full((B.shape[1], Y.shape[1]), np.nan)
    for c in range(Y.shape[1]):
        w = W[:, c] > 0
        if np.any(np.histogram(x[w], edges)[0] < min_per_span) or np.linalg.cond(B[w]) > MAX_COND:
            continue   # under-populated span or near-singular basis (Schoenberg-Whitney): the fit would be unbounded between stars
        C[:, c] = np.linalg.lstsq(B[w], Y[w, c], rcond=None)[0]
    f = BSpline(t, C, K, extrapolate=False)(grid)                                  # one spline, (G, 5)
    for c in range(Y.shape[1]):   # evaluate only where this column has data: grid inside [min, max] phi1 of its stars
        xc = x[W[:, c] > 0]
        if len(xc):
            f[(grid < xc.min()) | (grid > xc.max()), c] = np.nan
    return f


def fit_many(X, Y, W, grid, t):
    return np.stack([fit_eval(x, y, w, grid, t) for x, y, w in zip(X, Y, W)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_ou24vc_hydrabflow/training_data_300000.npz")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--noise", action=argparse.BooleanOptionalAction, default=True, help="Gaia DR3 errors on the sims (training steps %s)" % (NOISE_STEPS,))
    ap.add_argument("--simulator", default="stream_agama_spray_massloss_ibata_m200c_v4"); ap.add_argument("--noise-preset", default="stream_global")
    ap.add_argument("--n-interior", type=int, default=None, help="interior knots (default: the old rule clip(N_real // 25, 1, 6))")
    ap.add_argument("--min-sim-cov", type=float, default=0.5, help="drop grid points reached by fewer than this fraction of the sims")
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
        Wr = np.ones((N, 5)); Wr[:, 4] = rv
        t = knots(Fr[:, 0], a.n_interior if a.n_interior is not None else int(np.clip(N // 25, 1, 6))); real_t = np.unique(t)
        real_f = fit_eval(Fr[:, 0], np.nan_to_num(Fr[:, 1:]), Wr, grid, t)
        bi = rng.integers(0, N, (a.n_boot, N))   # real-curve uncertainty: star bootstrap (Gaia noise is in the members)
        boot = fit_many(Fr[bi, 0], np.nan_to_num(Fr[:, 1:])[bi], Wr[bi], grid, t)
        sig_real = np.nanstd(boot, 0)

        # draw rows of this stream until n_sim of them hold >= N stored in-window stars
        cand = rng.permutation(np.flatnonzero(j_all == j)); X, Y, W, used, tried = [], [], [], [], 0
        for chunk in np.array_split(cand, len(cand) // 500):
            rows = read_rows(a.sim, "sim_data_projected", np.sort(chunk)); tried += len(chunk)
            for r, st in zip(np.sort(chunk), rows):
                st = st[np.all(np.isfinite(st), 1) & (st[:, 0] > -900)].astype(float)
                if len(st) < N:
                    continue
                st = st[rng.choice(len(st), N, replace=False)]; st[:, 2] = 1.0 / st[:, 2]   # distance -> parallax
                w = np.ones((N, 5)); w[rng.choice(N, N - Nv, replace=False), 4] = 0
                X.append(st); W.append(w); used.append(r)
                if len(X) == a.n_sim: break
            if len(X) == a.n_sim: break
        ST = np.array(X)
        if noise is not None:
            ST = noise(ST, j)
        F = np.stack([to_frame(R, st) for st in ST]); X, Y = list(F[..., 0]), list(F[..., 1:])
        sim_f = fit_many(X, Y, W, grid, t)
        usable = np.isfinite(sim_f).any(1).mean(0)                                     # rows with a successful fit (per column)
        cov = np.isfinite(sim_f).mean(0)                                               # (G,5) fraction of sims reaching each grid point
        drop = (cov < a.min_sim_cov) | ~np.isfinite(real_f)   # compare only where BOTH real and (enough) sims have data
        sim_f[:, drop] = np.nan; real_f = np.where(drop, np.nan, real_f); sig_real = np.where(drop, np.nan, sig_real)

        fin = np.isfinite(sim_f)
        pct = np.sum(fin & (sim_f < real_f[None]), 0) / np.maximum(fin.sum(0), 1) * 100   # (G,5), finite sims only
        valid = np.isfinite(sim_f).mean(0)                                             # sim support covers grid point
        lo, hi = np.nanpercentile(sim_f, 5, 0), np.nanpercentile(sim_f, 95, 0)
        med = np.nanmedian(sim_f, 0); sig = 1.4826 * np.nanmedian(np.abs(sim_f - med), 0)
        rep[name] = dict(n_real=N, n_vlos=Nv, rows_tried=tried, rows_used=len(X),
                         frac_rows_with_enough_stars=len(X) / tried, usable_rows=usable.round(3).tolist(), real_knots=np.asarray(real_t).tolist(),
                         **{LABELS[c].split()[0]: dict(inside_5_95=float(np.mean(((real_f[:, c] >= lo[:, c]) & (real_f[:, c] <= hi[:, c]))[np.isfinite(real_f[:, c])])), n_grid_real=int(np.isfinite(real_f[:, c]).sum()),
                                                       median_pct=float(np.median(pct[np.isfinite(real_f[:, c]), c])), pct=pct[:, c].round(1).tolist(),
                                                       median_z=float(np.nanmedian((real_f[:, c] - med[:, c]) / np.hypot(sig[:, c], sig_real[:, c]))),
                                                       max_abs_z=float(np.nanmax(np.abs(real_f[:, c] - med[:, c]) / np.hypot(sig[:, c], sig_real[:, c]))),
                                                       median_sig_sim=float(np.nanmedian(sig[:, c])), median_sig_real=float(np.nanmedian(sig_real[:, c])),
                                                       sim_support=valid[:, c].round(3).tolist()) for c in range(5)})
        res[name] = (Fr, rv, grid, real_f, sig_real, sim_f, np.array(X), np.array(Y), np.array(W))
        print(name, json.dumps({k: v for k, v in rep[name].items() if k not in ("real_knots",)}, default=str)[:900])

    fig, ax = plt.subplots(5, 3, figsize=(15, 16), squeeze=False)
    for col, (name, (Fr, rv, grid, real_f, sig_real, sim_f, X, Y, W)) in enumerate(res.items()):
        for c in range(5):
            A = ax[c, col]; ok = rv if c == 4 else slice(None)
            A.scatter(X[:20].ravel()[W[:20, :, c].ravel() > 0], Y[:20, :, c].ravel()[W[:20, :, c].ravel() > 0], s=2, c="0.6", alpha=0.3, label="sim stars (20 rows)")
            for q, al in ((5, .2), (16, .35)):
                A.fill_between(grid, np.nanpercentile(sim_f[..., c], q, 0), np.nanpercentile(sim_f[..., c], 100 - q, 0), color="C0", alpha=al, lw=0)
            A.plot(grid, np.nanmedian(sim_f[..., c], 0), "C0-", lw=1.5, label="sim spline median / 68 / 90 %")
            A.scatter(Fr[ok, 0], Fr[ok, c + 1], s=6, c="k", label="real members")
            A.plot(grid, real_f[:, c], "o-", c="C3", ms=3, lw=1.5, label="real spline (20 phi1) +- bootstrap 1 sigma")
            A.fill_between(grid, real_f[:, c] - sig_real[:, c], real_f[:, c] + sig_real[:, c], color="C3", alpha=0.25, lw=0)
            v = np.concatenate([np.nanpercentile(sim_f[..., c], [5, 95], 0).ravel(), real_f[:, c], Fr[ok, c + 1]])
            v = v[np.isfinite(v)]; A.set_ylim(v.min() - 0.1 * np.ptp(v), v.max() + 0.1 * np.ptp(v))
            A.set_xlim(grid[0] - 3, grid[-1] + 3)
            A.set_title(f"{name}  {LABELS[c].split()[0]}: inside {rep[name][LABELS[c].split()[0]]['inside_5_95']:.2f}", fontsize=9)
            if col == 0: A.set_ylabel(LABELS[c])
            if c == 4: A.set_xlabel("phi1 [deg] (STREAMFINDER frame)")
    ax[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Prior predictive: one joint cubic LSQ B-spline (real-quantile knots) phi1 -> 5 observables; %s" % ("+ Gaia DR3 noise, no window/count cut" if a.noise else "no observation model"), fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "ppc_joint_bspline.png"), dpi=120)
    rep["_config"] = vars(a); json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
