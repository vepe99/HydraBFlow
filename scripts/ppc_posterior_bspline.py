#!/usr/bin/env python
"""Posterior-predictive check of a global real-data run through the TRAINING B-spline augmentation.

Takes the draws behind ``real_global_vs_streams_corner.png`` of an ``evaluate composition=global`` real-data
run -- the pooled global posterior (``posterior.npz``) and every stream's own single-stream posterior
(``single_stream_posterior.npz``) -- re-simulates the three streams for N draws of each, pushes them through
the model's TRAINING observation model (its ``model_dir`` config, up to ``mask_vlos``) and then through the
very ``stream_bspline_grid`` step the network was trained on (batched, GPU if visible). The real members go
through the real preset's chain and the same step, so both sides are the network's own stream input.

Locals (progenitor phase space, mass, t_end) are NOT inferred at the global level: they are drawn from
their prior, as in ``ppc_posterior_summary_statistics``. Marginalized/identity globals come from the prior.

Outputs (``<run>/ppc_bspline/``):
  * ``ppc_bspline_tracks.png``  per stream (rows) x observable (cols): 5-95 % / 16-84 % bands + median of the
    augmentation's spline for the pooled posterior (blue) and the stream's own posterior (orange), the real
    spline (black) over the members, the 80 % fit core marked.
  * ``ppc_bspline_<stream>.png`` one figure per stream with individual realization curves.
  * ``ppc_bspline.json``        per stream x observable x source: fraction of the grid inside 5-95 %,
    median / max |z| with z = (real - sim median) / 1.4826 MAD(sim).
  * ``ppc_bspline_streams.npz`` the re-simulated streams (``--reuse`` skips the simulation).

  HYDRABFLOW_NUM_GPUS=1 .venv/bin/python scripts/ppc_posterior_bspline.py \
      --run outputs/Bsline/spray_p1e3_v4_core080_2modal/eval_real --n-samples 100 --n-workers 48
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
# GPU whenever one is visible (the imported ppc_* helpers otherwise default JAX to CPU)
os.environ.setdefault("JAX_PLATFORMS", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_particle_coverage import real_clouds  # noqa: E402
from ppc_summary_statistics import augment_sim  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS = ["phi2 [deg]", "parallax [mas]", "mu_phi1 [mas/yr]", "mu_phi2 [mas/yr]", "v_los [km/s]"]
SRC = {"pooled": ("#1f3a93", "pooled global posterior"), "stream": ("#d9730d", "single-stream posterior")}


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def resimulate(args, sim, log10_keys, streams):
    """{source: (n, m, P, 6)} for source in pooled / stream (row s of 'stream' uses stream s's posterior)."""
    from hydrabflow.simulators.stream_common import sample_prior_value

    post = {"pooled": np.load(os.path.join(args.run, "posterior.npz")),
            "stream": np.load(os.path.join(args.run, "single_stream_posterior.npz"))}
    m, rng, out = len(streams), np.random.default_rng(args.seed), {}
    for src, d in post.items():
        n_draws = d[d.files[0]].shape[1]
        n = min(args.n_samples, n_draws)
        idx = rng.choice(n_draws, size=n, replace=False)
        flat = {}
        for key, spec in sim._priors_global.items():
            if key in d.files:
                v = np.asarray(d[key])[:, idx, 0]                      # (1 or m, n)
                v = 10.0 ** v if key in log10_keys else v              # posterior.npz is in native (log10) space
                flat[key] = np.broadcast_to(v, (m, n)).T.reshape(n * m, 1)
            elif spec["type"] == "identity":
                flat[key] = np.full((n * m, 1), float(spec["prior_parameters"][0]))
            else:
                flat[key] = np.repeat(sample_prior_value(spec, n, rng), m, axis=0)
        for key in sorted({k for name, _ in streams for k in sim._priors_local[name]}):
            flat[key] = np.concatenate([sample_prior_value(sim._priors_local[name][key], n, rng) for name, _ in streams],
                                       axis=1).reshape(n * m, 1)
        flat["j"] = np.tile(np.array([jj for _, jj in streams], float), n).reshape(n * m, 1)
        print(f"[{src}] re-simulating {n} draws x {m} streams ({args.n_particles} particles, {args.n_workers} workers)")
        res = sim.simulate(flat, np.random.default_rng(args.seed + 1))
        out[src] = res["sim_data_projected"].reshape(n, m, -1, 6)
    np.savez(os.path.join(args.out, "ppc_bspline_streams.npz"), **{f"{k}_sim_data_projected": v for k, v in out.items()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="evaluate composition=global real-data run dir")
    ap.add_argument("--n-samples", type=int, default=100, help="posterior draws per source")
    ap.add_argument("--n-particles", type=int, default=1000, help="1000 = the p1e3 training set")
    ap.add_argument("--n-workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-curves", type=int, default=60, help="individual curves drawn in the per-stream figures")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--simulator-preset", default="stream_agama_spray_massloss_ibata_m200c_v4", help="composes --real-aug")
    ap.add_argument("--real-aug", default="stream_real_global_streamfinder_bspline_core080", help="real-data augmentation preset")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from hydrabflow.pipeline.compositional import log10_keys_from_pipeline
    from hydrabflow.registry import AUGMENTATIONS, build_pipeline, get_simulator

    args.run = _abs(args.run)
    args.out = args.out or os.path.join(args.run, "ppc_bspline")
    os.makedirs(args.out, exist_ok=True)
    cfg = OmegaConf.load(os.path.join(args.run, ".hydra", "config.yaml"))
    cfg.simulator.params.n_particles, cfg.simulator.params.n_workers = args.n_particles, args.n_workers
    sim = get_simulator(cfg.simulator)
    streams = sorted(sim.target_streams.items(), key=lambda kv: kv[1])
    train_cfg = OmegaConf.load(os.path.join(_abs(str(cfg.model_dir)), ".hydra", "config.yaml"))
    log10_keys = set(log10_keys_from_pipeline(build_pipeline(train_cfg.preprocessing)))
    print(f"log10 (native-space) posterior keys: {sorted(log10_keys)}")

    cache = os.path.join(args.out, "ppc_bspline_streams.npz")
    if args.reuse and os.path.exists(cache):
        z = np.load(cache)
        grouped = {k.split("_")[0]: z[k] for k in z.files}
        print(f"reusing {cache}")
    else:
        grouped = resimulate(args, sim, log10_keys, streams)

    # the model's TRAINING augmentation: observation model up to mask_vlos, then its stream_bspline_grid step
    aug_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg.augmentation, resolve=True))
    aug_cfg.params["resources_dir"] = os.path.join(_REPO, "assets", "gaia")
    gp = OmegaConf.create(OmegaConf.to_container(aug_cfg.params, resolve=True))
    grid_fn = AUGMENTATIONS.get("stream_bspline_grid")(gp, np.random.default_rng(args.seed))
    key = str(gp.get("summary_key", "sim_summary"))

    def grid(stars, att, vm, j):
        b = {"sim_data_projected": np.asarray(stars, np.float32), "attention_mask": np.asarray(att, np.float32)[:, None],
             "vlos_mask": np.asarray(vm, np.float32)[:, None], "j": np.asarray(j, np.float32).reshape(-1, 1)}
        o = np.asarray(grid_fn(b)[key])                                   # (n, G, 9)
        vals = np.where(o[..., 5:6] > 0, o[..., :4], np.nan)             # invalid knot span -> not a datum
        return np.concatenate([vals, np.where(o[..., 6:7] > 0, o[..., 4:5], np.nan)], -1), o[..., 8]

    import jax
    print(f"[aug] training observation model + stream_bspline_grid of {cfg.model_dir} on {jax.devices()[0]}")
    curves = {}
    for src, g in grouped.items():
        n, m = g.shape[:2]
        jg = np.tile(np.arange(m, dtype=float), (n, 1))
        noisy, attn, vmask = augment_sim(g, jg, seed=args.seed, aug_cfg=aug_cfg)
        v, _ = grid(noisy.reshape(n * m, *noisy.shape[2:]), attn.reshape(n * m, -1), vmask.reshape(n * m, -1), jg.reshape(-1))
        ok = attn.reshape(n * m, -1).sum(1) >= 8
        v[~ok] = np.nan
        curves[src] = v.reshape(n, m, *v.shape[1:])                     # (n, m, G, 5)

    # real members: the real preset's chain up to impute_vlos, then the same grid step
    real_path = _abs(str(cfg.data.real_data_path))
    real = real_clouds(real_path, args.simulator_preset, args.real_aug, 100000, args.seed)
    from hydrabflow.augmentation.stream_bspline import _np_project, _stream_frames
    from hydrabflow.augmentation.stream_summary import _DEFAULT_CHANNELS
    frames = _stream_frames(gp, int(gp.bspline_interior_knots) + 1, int(gp.bspline_vlos_interior_knots) + 1, dict(_DEFAULT_CHANNELS))
    rv, rgrid, rstars = {}, {}, {}
    for name, j in streams:
        st, vm = real[j]
        v, gx = grid(st[None], np.ones((1, len(st))), vm[None], [j])
        rv[name], rgrid[name] = v[0], gx[0]
        F = np.column_stack(_np_project(frames.R[j], st.astype(float), dict(_DEFAULT_CHANNELS)))
        F[~vm, 5] = np.nan
        rstars[name] = F
    ev = frames.vlos_edges

    # ---- report + figures ----
    report, m = {}, len(streams)
    fig, axes = plt.subplots(m, 5, figsize=(21, 3.4 * m), squeeze=False)
    for s, (name, j) in enumerate(streams):
        report[name] = {}
        gx_t = rgrid[name]
        gx_v = np.linspace(ev[j, 0], ev[j, -1], len(gx_t))
        for c, lab in enumerate(OBS):
            ax, gx = axes[s, c], (gx_v if c == 4 else gx_t)
            real_c = rv[name][:, c]
            for src, (col, lbl) in SRC.items():
                C = curves[src][:, s, :, c]
                C = C[np.isfinite(C).all(1)]
                if len(C) < 5:
                    continue
                q5, q16, q50, q84, q95 = np.percentile(C, [5, 16, 50, 84, 95], 0)
                ax.fill_between(gx, q5, q95, color=col, alpha=0.13, lw=0)
                ax.fill_between(gx, q16, q84, color=col, alpha=0.25, lw=0)
                ax.plot(gx, q50, color=col, lw=1.4, label=f"{lbl} ({len(C)})")
                sig = 1.4826 * np.median(np.abs(C - q50), 0)
                z = (real_c - q50) / np.where(sig > 0, sig, np.nan)
                report[name].setdefault(lab, {})[src] = dict(
                    n_usable=int(len(C)), inside_5_95=float(np.nanmean((real_c >= q5) & (real_c <= q95))),
                    median_z=float(np.nanmedian(z)), max_abs_z=float(np.nanmax(np.abs(z))))
            F = rstars[name]
            okr = np.isfinite(F[:, c + 1])
            ax.scatter(F[okr, 0], F[okr, c + 1], s=5, c="0.35", alpha=0.45, lw=0)
            ax.plot(gx, real_c, color="k", lw=2.0, label="Gaia members (same spline)")
            r = report[name].get(lab, {})
            ax.set_title("  ".join(f"{k}: in {v['inside_5_95']:.2f} z {v['median_z']:+.2f}" for k, v in r.items()), fontsize=8)
            lo, hi = np.nanpercentile(np.r_[real_c, F[okr, c + 1]], [1, 99])
            pad = 0.45 * (hi - lo) + 1e-3
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_ylabel(lab)
            ax.grid(alpha=0.25)
            if c == 0:
                ax.text(0.02, 0.95, name, transform=ax.transAxes, fontweight="bold", va="top")
            if s == m - 1:
                ax.set_xlabel("phi1 [deg]  (STREAMFINDER frame)")
    axes[0, -1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Posterior-predictive check through the training B-spline augmentation (stream_bspline_grid, core080): "
                 "re-simulated streams vs Gaia members", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ppc_bspline_tracks.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    for s, (name, j) in enumerate(streams):          # per-stream figure with individual realizations
        fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
        gx_t = rgrid[name]
        gx_v = np.linspace(ev[j, 0], ev[j, -1], len(gx_t))
        F = rstars[name]
        for c, (ax, lab) in enumerate(zip(axes, OBS)):
            gx = gx_v if c == 4 else gx_t
            for src, (col, lbl) in SRC.items():
                C = curves[src][:, s, :, c]
                C = C[np.isfinite(C).all(1)][: args.max_curves]
                for k, cv in enumerate(C):
                    ax.plot(gx, cv, color=col, alpha=0.18, lw=0.8, label=lbl if k == 0 else None)
            okr = np.isfinite(F[:, c + 1])
            ax.scatter(F[okr, 0], F[okr, c + 1], s=7, c="0.3", alpha=0.6, lw=0, label="Gaia members")
            ax.plot(gx, rv[name][:, c], color="k", lw=2.2, label="Gaia spline")
            lo, hi = np.nanpercentile(F[okr, c + 1], [2, 98])
            ax.set_ylim(lo - 0.6 * (hi - lo), hi + 0.6 * (hi - lo))
            ax.set_ylabel(lab)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7, loc="upper right")
        axes[-1].set_xlabel("phi1 [deg]  (STREAMFINDER frame)")
        axes[0].set_title(f"{name}: augmentation splines of {args.max_curves} realizations per posterior source")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"ppc_bspline_{name}.png"), dpi=120)
        plt.close(fig)

    json.dump(dict(run=args.run, n_samples=args.n_samples, n_particles=args.n_particles, report=report),
              open(os.path.join(args.out, "ppc_bspline.json"), "w"), indent=1)
    print(f"\n{'stream':8s} {'observable':18s} {'source':7s} {'usable':>6s} {'inside':>6s} {'z_med':>6s} {'|z|max':>6s}")
    for name, d in report.items():
        for lab, r in d.items():
            for src, v in r.items():
                print(f"{name:8s} {lab:18s} {src:7s} {v['n_usable']:6d} {v['inside_5_95']:6.2f} {v['median_z']:+6.2f} {v['max_abs_z']:6.2f}")
    print(f"\nartifacts in {args.out}")


if __name__ == "__main__":
    main()
