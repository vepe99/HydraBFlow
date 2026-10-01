#!/usr/bin/env python
"""Prior-predictive corner plots in observation space: ALL simulated in-window stars vs Gaia members.

One corner per stream over the catalogue astrometry (alpha, delta, parallax, mu_alpha*, mu_delta), no
diagonal and no v_los. Every simulated particle inside the stream's RA/Dec observation window is drawn
through the training observation model (G magnitude, DR3 sigma(G) errors) but WITHOUT the member-count
subsample, i.e. every star, not the observed-size subset the network trains on. Real members
through their preset. Prior stars use the stream colours of the eval_real global-vs-streams corner.

  .venv/bin/python scripts/ppc_obs_corner.py --out outputs/Bsline/streamfinder_spray_p1e3_v4_prog2026/ppc_stream_aug_core080
"""
import argparse, os, sys
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0"); os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import NAMES  # noqa: E402
from ppc_particle_coverage import compose_aug, real_clouds  # noqa: E402
from ppc_observation_space import load_sim_groups  # noqa: E402

LABELS = [r"$\alpha$ [deg]", r"$\delta$ [deg]", r"$\pi$ [mas]", r"$\mu_{\alpha*}$ [mas/yr]",
          r"$\mu_\delta$ [mas/yr]", r"$v_{\rm los}$ [km/s]"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow/training_data_300000.npz")
    ap.add_argument("--simulator", default="stream_agama_spray_massloss_ibata_m200c_v4")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--aug", default="stream_global_streamfinder_bspline_core080")
    ap.add_argument("--real-aug", default="stream_real_global_streamfinder_bspline_core080")
    ap.add_argument("--n-sim", type=int, default=50000, help="realizations per stream whose stars are pooled")
    ap.add_argument("--n-points", type=int, default=10000, help="prior stars plotted, after the box cut (0 = all)")
    ap.add_argument("--batch", type=int, default=2000, help="rows per augmentation call")
    ap.add_argument("--alpha", type=float, default=0.0, help="prior point alpha; 0 = 1e5/N clipped to [0.01, 0.35]"); ap.add_argument("--ms", type=float, default=2.0)
    ap.add_argument("--label", default="Prior", help="legend label of the simulated stars (e.g. Posterior)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    real = real_clouds(a.real, a.simulator, a.real_aug, 1000, a.seed)
    sim_raw, jj = load_sim_groups(a.sim, a.n_sim, rng)
    # the training observation model WITHOUT the member-count subsample: every in-window star gets a
    # G magnitude + DR3 sigma(G) errors and is perturbed, exactly as in training
    from omegaconf import OmegaConf
    from hydrabflow.registry import build_augmentations
    acfg = OmegaConf.create(OmegaConf.to_container(compose_aug(a.simulator, a.aug), resolve=True))
    acfg.steps = ["convert_distance_to_parallax", "observational_window", "compact_to_attended",
                  "sample_magnitudes", "sample_obs_error", "apply_obs_error"]
    acfg.params.max_particles = int(sim_raw.shape[2])
    if not os.path.exists(os.path.join(str(acfg.params.resources_dir), str(acfg.params.member_table))):
        acfg.params.resources_dir = "assets/gaia"
    chain = build_augmentations(acfg, np.random.default_rng(a.seed), context={})

    cols = plt.cm.RdYlBu_r(np.linspace(0, 1, 4))[1:]            # the eval_real corner colours (0 = Global)
    sky = {}
    for j, name in NAMES.items():
        parts = []
        for i in range(0, sim_raw.shape[0], a.batch):
            b = {"sim_data_projected": np.asarray(sim_raw[i:i + a.batch, j], np.float32),
                 "j": np.full((len(sim_raw[i:i + a.batch]), 1), j, np.float32)}
            for fn in chain:
                b = fn(b)
            st, at = np.asarray(b["sim_data_projected"]), np.asarray(b["attention_mask"])[:, 0].astype(bool)
            parts.append(st[at][:, :5].astype(float))
        X = np.concatenate(parts); X = X[np.isfinite(X).all(1)]
        Y = np.asarray(real[j][0], float)[:, :5]
        w = acfg.params.observational_window[name]          # same RA/Dec window as the prior stars
        inw = (Y[:, 0] >= w.ra_min) & (Y[:, 0] <= w.ra_max) & (Y[:, 1] >= w.dec_min) & (Y[:, 1] <= w.dec_max)
        print(f"[{name}] real members: {inw.sum()} of {len(Y)} inside the observational window")
        Y = Y[inw]
        print(f"[{name}] {len(X)} in-window simulated stars from {sim_raw.shape[0]} draws")
        # second cut: the box spanned by the (windowed) Gaia members in every observable, then one
        # random subset of --n-points prior stars shared by all panels
        lo, hi = Y.min(0), Y.max(0)
        X = X[((X >= lo) & (X <= hi)).all(1)]
        print(f"[{name}] {len(X)} simulated stars inside the members' box")
        if 0 < a.n_points < len(X):
            X = X[rng.choice(len(X), a.n_points, replace=False)]
        al = a.alpha if a.alpha > 0 else float(np.clip(1e5 / max(len(X), 1), 0.01, 0.35))   # 0 = scale with the star count
        rng_ = list(zip(lo, hi))  # the stars are cut to this box: axes end exactly at it, no padding

        fig, axs = plt.subplots(4, 4, figsize=(11, 11))
        for r in range(1, 5):
            for c in range(4):
                ax = axs[r - 1, c]
                if c >= r:
                    ax.axis("off"); continue
                ax.plot(X[:, c], X[:, r], ".", ms=a.ms, color=cols[j], alpha=al, mew=0, rasterized=True)
                ax.plot(Y[:, c], Y[:, r], "o", ms=2.6, color="k", alpha=0.85, mew=0, zorder=10)
                ax.set_xlim(rng_[c]); ax.set_ylim(rng_[r])
                if r == 4:
                    ax.set_xlabel(LABELS[c])
                else:
                    ax.set_xticklabels([])
                if c == 0:
                    ax.set_ylabel(LABELS[r])
                else:
                    ax.set_yticklabels([])
        h = [plt.Line2D([], [], ls="", marker="o", color=cols[j]), plt.Line2D([], [], ls="", marker="o", color="k")]
        fig.legend(h, [a.label, "Gaia"],
                   loc="upper right", bbox_to_anchor=(0.9, 0.86), fontsize=13, frameon=False)
        fig.subplots_adjust(hspace=0.06, wspace=0.06)
        p = os.path.join(a.out, f"ppc_corner_{name}.pdf")
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        print("wrote", p)

        # sky-only corner (alpha, delta) WITH the 1-D marginals on the diagonal
        fig, axs = plt.subplots(2, 2, figsize=(6.5, 6.5))
        sky_corner(axs, X, Y, rng_, cols[j], al, a.ms)
        fig.legend(h, [a.label, "Gaia"], loc="upper right", bbox_to_anchor=(0.9, 0.86), fontsize=12, frameon=False)
        fig.subplots_adjust(hspace=0.06, wspace=0.06)
        p = os.path.join(a.out, f"ppc_corner_sky_{name}.pdf")
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        print("wrote", p)
        sky[name] = (X, Y, rng_, cols[j], al, h)

    # the three sky corners side by side (3:1), bigger labels, delta label only on the first
    order = [n for n in ("M68", "NGC3201", "Pal5") if n in sky]
    fig = plt.figure(figsize=(6.5 * len(order), 6.5))
    for sf, name in zip(fig.subfigures(1, len(order), wspace=0), order):
        X, Y, rng_, c, al, h = sky[name]
        first = name == order[0]  # only the first corner carries the delta label -> wider left margin
        axs = sf.subplots(2, 2, gridspec_kw={"hspace": 0.06, "wspace": 0.06, "left": 0.14 if first else 0.11,
                                             "right": 0.99})
        sky_corner(axs, X, Y, rng_, c, al, a.ms, fs=22, ylabel=first)
        axs[0, 1].legend(h, [a.label, "Gaia"], loc="center", fontsize=18, frameon=False, title=name,
                         title_fontsize=18)
    p = os.path.join(a.out, "ppc_corner_sky_all.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight"); fig.savefig(p[:-4] + ".png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def sky_corner(axs, X, Y, rng_, color, al, ms, fs=None, ylabel=True):
    """(alpha, delta) scatter + 1-D marginals of simulated stars X vs members Y on a 2x2 grid."""
    for k in range(2):
        ax = axs[k, k]
        ax.hist(X[:, k], bins=60, range=rng_[k], density=True, histtype="stepfilled", color=color, alpha=0.6, lw=0)
        ax.hist(Y[:, k], bins=30, range=rng_[k], density=True, histtype="step", color="k", lw=1.4)
        ax.set_xlim(rng_[k]); ax.set_yticks([])
    ax = axs[1, 0]
    ax.plot(X[:, 0], X[:, 1], ".", ms=ms, color=color, alpha=al, mew=0, rasterized=True)
    ax.plot(Y[:, 0], Y[:, 1], "o", ms=2.6, color="k", alpha=0.85, mew=0, zorder=10)
    ax.set_xlim(rng_[0]); ax.set_ylim(rng_[1])
    ax.set_xlabel(LABELS[0], fontsize=fs)
    if ylabel:
        ax.set_ylabel(LABELS[1], fontsize=fs)
    axs[1, 1].set_xlabel(LABELS[1], fontsize=fs)
    axs[0, 0].set_xticklabels([]); axs[0, 1].axis("off")


if __name__ == "__main__":
    main()
