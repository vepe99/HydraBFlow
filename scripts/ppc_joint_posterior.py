"""Joint (global x local) posterior-predictive check in observation space.

Takes the PAIRED draws of a hierarchical real-data run -- the global posterior of an
``evaluate composition=global`` run and the local posterior an ``evaluate composition=local``
run drew from it by ancestral sampling (draw k of the locals was conditioned on draw k of the
globals, so row k is a joint sample of the whole model) -- re-simulates each (global, local)
pair with the run's own simulator, pushes the streams through the TRAINING observation model
(window, member-count subsample, Gaia magnitudes/errors, v_los mask) and compares them with the
real Gaia members in the published STREAMFINDER phi1/phi2 frames:

* ``ppc_joint_scatter.png``  -- per stream, the five observables (phi2, parallax, mu_phi1, mu_phi2,
  v_los) against phi1: simulated attended stars from all draws (light) under the real members (dark).
* ``ppc_joint_bspline.png``  -- per stream x observable, the smoothing B-spline track of every
  re-simulated realization (median + 5-95 % band) against the real members' track, on the same
  knot vector (the ``--fit smoothing`` recipe of ppc_bspline_nn: GCV lambda from the real fit, held
  fixed for the sims). Fraction of the grid where the real curve sits inside the band, and the
  median z = (real - sim median) / 1.4826 MAD(sim), per stream x observable -> ``ppc_joint.json``.

Usage:
  HYDRABFLOW_NUM_GPUS=0 .venv/bin/python scripts/ppc_joint_posterior.py \
      --local-run outputs/Bsline/spray_p1e3_v4_smoothing_local/eval_real \
      --global-run outputs/Bsline/spray_p1e3_v4_smoothing_2modal/eval_real \
      --n-samples 40 --n-particles 1000 --n-workers 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppc_bspline_nn as bsp  # noqa: E402
from ppc_particle_coverage import real_clouds, to_frame  # noqa: E402
from ppc_summary_statistics import augment_sim  # noqa: E402
from streamfinder_frame import frames as streamfinder_frames  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS = [("phi2", "phi2 [deg]", 1), ("parallax", "parallax [mas]", 2), ("mu_phi1", "mu_phi1 [mas/yr]", 3),
       ("mu_phi2", "mu_phi2 [mas/yr]", 4), ("vlos", "v_los [km/s]", 5)]


def _post(path):
    d = np.load(path, allow_pickle=True)
    return {k: np.asarray(d[k]) for k in d.files}


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def resimulate(args, cfg, sim, log10_keys):
    """Flat draw-major rows (n*m) of paired (global, local) draws -> simulate -> (n, m, P, 6)."""
    from hydrabflow.simulators.stream_common import sample_prior_value

    gpost = _post(os.path.join(args.global_run, "posterior.npz"))
    lpost = _post(os.path.join(args.local_run, "posterior.npz"))
    for k in set(gpost) & log10_keys:                    # global posterior.npz is in native (log10) space
        gpost[k] = 10.0 ** gpost[k]
    streams = sorted(sim.target_streams.items(), key=lambda kv: kv[1])
    m = len(streams)
    n_draws = next(iter(gpost.values())).shape[1]
    lshape = next(iter(lpost.values())).shape           # (1, m, n_parent, 1)
    assert lshape[1] == m and lshape[2] == n_draws, f"local posterior {lshape} does not pair with {n_draws} global draws"
    n = min(args.n_samples, n_draws)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_draws, size=n, replace=False)

    flat = {}
    for key, spec in sim._priors_global.items():
        if key in gpost:
            flat[key] = np.repeat(gpost[key][0, idx].reshape(n, 1), m, axis=0)
        elif spec["type"] == "identity":
            flat[key] = np.full((n * m, 1), float(spec["prior_parameters"][0]))
        else:                                             # marginalized nuisance (solar frame): prior
            flat[key] = np.repeat(sample_prior_value(spec, n, rng), m, axis=0)
    from_local = []
    for key in sorted({k for name, _ in streams for k in sim._priors_local[name]}):
        cols = []
        for s, (name, _) in enumerate(streams):
            if key in lpost:
                cols.append(lpost[key][0, s, idx].reshape(n, 1))
            else:
                cols.append(sample_prior_value(sim._priors_local[name][key], n, rng))
        if key in lpost:
            from_local.append(key)
        flat[key] = np.concatenate(cols, axis=1).reshape(n * m, 1)
    flat["j"] = np.tile(np.array([jj for _, jj in streams], dtype=float), n).reshape(n * m, 1)
    print(f"re-simulating {n} joint draws x {m} streams with {type(sim).__name__} "
          f"({cfg.simulator.params.n_particles} particles, {cfg.simulator.params.n_workers} workers)")
    print(f"  globals from {args.global_run}: {[k for k in sim._priors_global if k in gpost]}")
    print(f"  locals  from {args.local_run}: {from_local}")
    out = sim.simulate(flat, np.random.default_rng(args.seed + 1))
    grouped = out["sim_data_projected"].reshape(n, m, -1, 6)
    nan_rows = int(np.isnan(grouped).all(axis=(2, 3)).sum())
    if nan_rows:
        print(f"  {nan_rows}/{n * m} re-simulated streams are NaN (failed rows)")
    np.savez(os.path.join(args.out, "ppc_joint_streams.npz"), sim_data_projected=grouped,
             **{k: v.reshape(n, m) for k, v in flat.items()})
    return grouped, streams


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-run", required=True, help="evaluate composition=local real-data run dir")
    ap.add_argument("--global-run", required=True, help="evaluate composition=global real-data run dir")
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--n-particles", type=int, default=1000)
    ap.add_argument("--n-workers", type=int, default=24)
    ap.add_argument("--real", default=None, help="real member npz (default: the local run's data.real_data_path)")
    ap.add_argument("--real-aug", default="stream_real_global_streamfinder_bspline_smoothing",
                    help="real-data augmentation preset (composed with --simulator-preset for the real members)")
    ap.add_argument("--simulator-preset", default="stream_agama_spray_massloss_ibata_m200c_v4",
                    help="conf/simulator preset name used to compose --real-aug")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-scatter", type=int, default=4000, help="sim stars drawn per panel (plot size)")
    ap.add_argument("--out", default=None, help="output dir (default: <local-run>/ppc_joint)")
    ap.add_argument("--fit", default="pspline", choices=("pspline", "smoothing"),
                    help="pspline = the TRAINING estimator (stream_bspline_grid bspline_fit=smoothing): uniform knots + "
                         "2nd-difference penalty, knot range/occupancy knots and lambda from the training real_streams_file; "
                         "smoothing = scipy make_smoothing_spline with GCV lambda on the plotted members")
    ap.add_argument("--lam", type=float, default=None,
                    help="pin the spline lambda (real + every sim) instead of the GCV value; output files get a _lam<value> suffix")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse <out>/ppc_joint_streams.npz from a previous run instead of re-simulating")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from hydrabflow.pipeline.compositional import log10_keys_from_pipeline
    from hydrabflow.registry import build_pipeline, get_simulator

    args.local_run, args.global_run = _abs(args.local_run), _abs(args.global_run)
    args.out = args.out or os.path.join(args.local_run, "ppc_joint")
    os.makedirs(args.out, exist_ok=True)
    cfg = OmegaConf.load(os.path.join(args.local_run, ".hydra", "config.yaml"))
    cfg.simulator.params.n_particles = args.n_particles
    cfg.simulator.params.n_workers = args.n_workers
    sim = get_simulator(cfg.simulator)
    log10_keys = set(log10_keys_from_pipeline(build_pipeline(cfg.preprocessing)))
    real_path = _abs(args.real or str(cfg.data.real_data_path))

    cache = os.path.join(args.out, "ppc_joint_streams.npz")
    if args.reuse and os.path.exists(cache):
        grouped = np.load(cache)["sim_data_projected"]
        streams = sorted(sim.target_streams.items(), key=lambda kv: kv[1])
        print(f"reusing {grouped.shape[0]} re-simulated joint draws from {cache}")
    else:
        grouped, streams = resimulate(args, cfg, sim, log10_keys)
    n, m = grouped.shape[:2]

    # TRAINING observation model of the local model (its model_dir's config), up to mask_vlos.
    train_cfg = OmegaConf.load(os.path.join(_abs(str(cfg.model_dir)), ".hydra", "config.yaml"))
    aug_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg.augmentation, resolve=True))
    jgrid = np.tile(np.array([jj for _, jj in streams], dtype=float), (n, 1))
    noisy, attn, vmask = augment_sim(grouped, jgrid, seed=args.seed, aug_cfg=aug_cfg)

    real = real_clouds(real_path, args.simulator_preset, args.real_aug, 100000, args.seed)
    frames = streamfinder_frames()
    # pspline: knots + lambda come from the members the training augmentation was built on
    ref_path = _abs(str(aug_cfg.params.real_streams_file))
    ref = real if ref_path == real_path else real_clouds(ref_path, args.simulator_preset, args.real_aug, 100000, args.seed)

    # ---------- per stream: project sims + real into the published frame ----------
    sim_tab, real_tab, ref_tab = {}, {}, {}
    for s, (name, j) in enumerate(streams):
        R = frames[name]
        rows = []
        for i in range(n):
            st = noisy[i, s][attn[i, s]]
            if len(st) == 0 or not np.isfinite(st).all():
                rows.append(None)
                continue
            F = to_frame(R, st.astype(float))
            F[~vmask[i, s][attn[i, s]], 5] = np.nan       # unmeasured v_los -> not a datum
            rows.append(F)
        sim_tab[name] = rows
        rst, rvm = real[j]
        F = to_frame(R, rst.astype(float))
        F[~rvm, 5] = np.nan
        real_tab[name] = F
        rst, rvm = ref[j]
        F = to_frame(R, rst.astype(float))
        F[~rvm, 5] = np.nan
        ref_tab[name] = F

    # ---------- figure 1: scatter ----------
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(m, len(OBS), figsize=(4.2 * len(OBS), 3.4 * m), squeeze=False)
    for s, (name, j) in enumerate(streams):
        Fr = real_tab[name]
        Fs = np.concatenate([F for F in sim_tab[name] if F is not None], axis=0)
        for c, (key, lab, col) in enumerate(OBS):
            ax = axes[s, c]
            ok = np.isfinite(Fs[:, col])
            pick = np.flatnonzero(ok)
            if len(pick) > args.max_scatter:
                pick = rng.choice(pick, args.max_scatter, replace=False)
            ax.scatter(Fs[pick, 0], Fs[pick, col], s=4, c="#8fa6ff", alpha=0.35, lw=0, label=f"sim ({n} joint draws)")
            okr = np.isfinite(Fr[:, col])
            ax.scatter(Fr[okr, 0], Fr[okr, col], s=9, c="#1d2230", alpha=0.9, lw=0, label="Gaia members")
            lo, hi = np.nanpercentile(np.r_[Fs[ok, col], Fr[okr, col]], [1, 99])
            pad = 0.25 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(*np.nanpercentile(Fr[:, 0], [0, 100]) + np.array([-3, 3]))
            ax.set_ylabel(lab if c == 0 or True else "")
            if s == m - 1:
                ax.set_xlabel("phi1 [deg]")
            if c == 0:
                ax.set_title(f"{name}", loc="left", fontweight="bold")
            if s == 0 and c == len(OBS) - 1:
                ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.25)
    fig.suptitle("Joint posterior-predictive check: re-simulated streams (training observation model) vs Gaia members, STREAMFINDER frames", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ppc_joint_scatter.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- figure 2: smoothing B-spline tracks ----------
    bsp.FIT = args.fit
    sfx = ("" if args.fit == "smoothing" else f"_{args.fit}") + ("" if args.lam is None else f"_lam{args.lam:g}")
    n_int = {"track": 5, "vlos": 1}
    report = {}
    fig, axes = plt.subplots(m, len(OBS), figsize=(4.2 * len(OBS), 3.4 * m), squeeze=False)
    for s, (name, j) in enumerate(streams):
        Fr = real_tab[name]
        report[name] = {}
        for c, (key, lab, col) in enumerate(OBS):
            ax = axes[s, c]
            kind = "vlos" if key == "vlos" else "track"
            okr = np.isfinite(Fr[:, col])
            if args.fit == "pspline":   # training: knots over the reference members, their GCV lambda applied to everyone
                Fref = ref_tab[name]
                okf = np.isfinite(Fref[:, col])
                t = bsp.knots(Fref[okf, 0], n_int[kind])
                grid = np.linspace(t[0], t[-1], 40)
                _, lam = bsp.spline_on_grid(Fref[okf, 0], Fref[okf, col], t, grid, lam=None, return_lam=True, kind=kind)
                print(f"  {name:8s} {key:9s} GCV lambda {lam:.3g}" + ("" if args.lam is None else f" -> pinned {args.lam:g}"))
                lam = lam if args.lam is None else args.lam
                real_curve = bsp.spline_on_grid(Fr[okr, 0], Fr[okr, col], t, grid, lam=lam, kind=kind, min_per_span=0)
            else:
                t = bsp.knots(Fr[okr, 0], n_int[kind])
                grid = np.linspace(t[0], t[-1], 40)
                real_curve, lam = bsp.spline_on_grid(Fr[okr, 0], Fr[okr, col], t, grid, lam=args.lam, return_lam=True, kind=kind)
            curves = []
            for F in sim_tab[name]:
                if F is None:
                    continue
                ok = np.isfinite(F[:, col])
                curves.append(bsp.spline_on_grid(F[ok, 0], F[ok, col], t, grid, lam=lam, kind=kind))
            C = np.array(curves) if curves else np.full((1, len(grid)), np.nan)
            usable = np.isfinite(C).all(axis=1)
            C = C[usable]
            for cv in C:
                ax.plot(grid, cv, color="#8fa6ff", alpha=0.25, lw=0.8)
            if len(C):
                q5, q50, q95 = np.nanpercentile(C, [5, 50, 95], axis=0)
                ax.fill_between(grid, q5, q95, color="#1f3a93", alpha=0.18, lw=0, label="sim 5-95 %")
                ax.plot(grid, q50, color="#1f3a93", lw=1.4, label="sim median")
                sig = 1.4826 * np.nanmedian(np.abs(C - q50), axis=0)
                z = (real_curve - q50) / np.where(sig > 0, sig, np.nan)
                inside = float(np.mean((real_curve >= q5) & (real_curve <= q95)))
                report[name][key] = dict(n_usable=int(len(C)), inside_5_95=inside,
                                         median_z=float(np.nanmedian(z)), max_abs_z=float(np.nanmax(np.abs(z))))
                ax.set_title(f"inside {inside:.2f}  z_med {np.nanmedian(z):+.2f}  ({len(C)} sims)", fontsize=9)
            ax.plot(grid, real_curve, color="#1d2230", lw=2.0, label="Gaia members")
            ax.scatter(Fr[okr, 0], Fr[okr, col], s=5, c="#1d2230", alpha=0.35, lw=0)
            if len(C):
                lo, hi = np.nanpercentile(np.r_[C.ravel(), real_curve], [0.5, 99.5])
                pad = 0.3 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_ylabel(lab)
            if s == m - 1:
                ax.set_xlabel("phi1 [deg]")
            if c == 0:
                ax.text(0.02, 0.95, name, transform=ax.transAxes, fontweight="bold", va="top")
            if s == 0 and c == len(OBS) - 1:
                ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.25)
    fig.suptitle(f"Joint posterior-predictive check: {args.fit} B-spline tracks of re-simulated streams vs the Gaia members' track", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"ppc_joint_bspline{sfx}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- figures 3/4: observation-space corner plots, 5-D (all stars) and 6-D (measured v_los) ----------
    import corner

    labels6 = ["phi1 [deg]", "phi2 [deg]", "parallax [mas]", "mu_phi1 [mas/yr]", "mu_phi2 [mas/yr]", "v_los [km/s]"]
    for s, (name, j) in enumerate(streams):
        Fs = np.concatenate([F for F in sim_tab[name] if F is not None], axis=0)
        Fr = real_tab[name]
        for dim, tag in ((5, "5d"), (6, "6d")):
            X, Y = Fs[:, :dim], Fr[:, :dim]
            if dim == 6:                                   # only stars with a measured v_los, both sides
                X, Y = X[np.isfinite(X[:, 5])], Y[np.isfinite(Y[:, 5])]
            X = X[np.isfinite(X).all(axis=1)]
            if len(X) > 60000:
                X = X[rng.choice(len(X), 60000, replace=False)]
            both = np.concatenate([X, Y], axis=0)
            rng_ = [tuple(np.nanpercentile(both[:, k], [0.5, 99.5]) + np.array([-1, 1]) * 0.08 * np.ptp(np.nanpercentile(both[:, k], [0.5, 99.5]))) for k in range(dim)]
            fig = corner.corner(X, labels=labels6[:dim], range=rng_, color="#1f3a93", bins=40,
                                levels=(0.68, 0.95), plot_datapoints=False, plot_density=True,
                                fill_contours=True, smooth=1.0, hist_kwargs=dict(density=True, lw=1.4))
            # corner pins data points at zorder=-1 (under the filled contours), so scatter the members by hand
            axs = np.array(fig.axes).reshape(dim, dim)
            for r in range(dim):
                axs[r, r].hist(Y[:, r], bins=40, range=rng_[r], density=True, histtype="step", color="#1d2230", lw=1.4)
                for c in range(r):
                    axs[r, c].plot(Y[:, c], Y[:, r], "o", ms=3.4, color="#1d2230", alpha=0.9, zorder=10, mew=0)
            fig.suptitle(f"{name}: simulated stars from {n} joint posterior draws (blue, 68/95 %) vs Gaia members (black)"
                         + (f"  --  {len(Y)} members / {len(X)} sim stars with measured v_los" if dim == 6 else
                            f"  --  {len(Y)} members / {len(X)} sim stars"), y=1.01, fontsize=11)
            fig.savefig(os.path.join(args.out, f"ppc_joint_corner{tag}_{name}.png"), dpi=110, bbox_inches="tight")
            plt.close(fig)
    print("corner plots written")

    with open(os.path.join(args.out, f"ppc_joint{sfx}.json"), "w") as f:
        json.dump(dict(n_draws=n, n_particles=args.n_particles, global_run=args.global_run,
                       local_run=args.local_run, real=real_path, bspline=report), f, indent=2)
    print(f"\n{'stream':9s} {'obs':9s} {'usable':>6s} {'inside':>7s} {'z_med':>7s} {'|z|max':>7s}")
    for name, d in report.items():
        for key, r in d.items():
            print(f"{name:9s} {key:9s} {r['n_usable']:6d} {r['inside_5_95']:7.2f} {r['median_z']:+7.2f} {r['max_abs_z']:7.2f}")
    print(f"\nartifacts in {args.out}")


if __name__ == "__main__":
    main()
