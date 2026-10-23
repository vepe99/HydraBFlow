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
from matplotlib.patches import Patch
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import NAMES  # noqa: E402
from ppc_particle_coverage import compose_aug, real_clouds  # noqa: E402
from ppc_observation_space import load_sim_groups  # noqa: E402
from plot_loo_ppc import density_levels  # noqa: E402

LINE_WIDTH = 2.4

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
    ap.add_argument("--overlay", default=None, help="second sim npz drawn as dashed 68/95 %% contours; "
                    "'{name}' is replaced by the stream name (e.g. a leave-one-out without_{name}/ppc_strict_n500)")
    ap.add_argument("--overlay-label", default="Consistency test")
    ap.add_argument("--prior", default=None, help="prior sim npz drawn as SOLID 68/95 %% contour lines (same --n-sim)")
    ap.add_argument("--prior-label", default="Prior")
    ap.add_argument("--line-color", default="#1b2a6b", help="colour of the --prior / --overlay lines; 'stream' = the "
                    "stream colour, 'stream-dark' = a darker shade of it (the posterior fill then drops its edge line, which would read as the prior)")
    ap.add_argument("--contours", action="store_true", help="draw the simulated stars as filled 68/95 %% contours "
                    "in the stream colour (from all box stars) instead of points -> ppc_corner_contours_<stream>")
    ap.add_argument("--fontsize", type=float, default=None, help="base font size of the stream corners (default: matplotlib's)")
    ap.add_argument("--dim", type=int, default=5, choices=(5, 6), help="6 adds v_los (members: measured v_los only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.fontsize:
        plt.rcParams.update({"font.size": a.fontsize, "axes.labelsize": a.fontsize * 1.15})
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

    def stars(raw, j):
        """In-window simulated stars of stream j, through the observation chain above."""
        parts = []
        for i in range(0, raw.shape[0], a.batch):
            b = {"sim_data_projected": np.asarray(raw[i:i + a.batch, j], np.float32),
                 "j": np.full((len(raw[i:i + a.batch]), 1), j, np.float32)}
            for fn in chain:
                b = fn(b)
            st, at = np.asarray(b["sim_data_projected"]), np.asarray(b["attention_mask"])[:, 0].astype(bool)
            parts.append(st[at][:, :a.dim].astype(float))
        X = np.concatenate(parts)
        return X[np.isfinite(X).all(1)]

    cols = plt.cm.RdYlBu_r(np.linspace(0, 1, 4))[1:]            # the eval_real corner colours (0 = Global)
    sky = {}
    for j, name in NAMES.items():
        X = stars(sim_raw, j)
        Y, vm = np.asarray(real[j][0], float)[:, :a.dim], np.asarray(real[j][1], bool)
        w = acfg.params.observational_window[name]          # same RA/Dec window as the prior stars
        inw = (Y[:, 0] >= w.ra_min) & (Y[:, 0] <= w.ra_max) & (Y[:, 1] >= w.dec_min) & (Y[:, 1] <= w.dec_max)
        print(f"[{name}] real members: {inw.sum()} of {len(Y)} inside the observational window")
        Y, vm = Y[inw], vm[inw]
        print(f"[{name}] {len(X)} in-window simulated stars from {sim_raw.shape[0]} draws")
        # second cut: the box spanned by the (windowed) Gaia members in every observable, then one
        # random subset of --n-points prior stars shared by all panels
        # the box is astrometric only; the v_los axis spans the MEASURED members and the contour
        # histograms simply ignore stars outside it (no cut on v_los, which would bias the other panels)
        lo, hi = Y[:, :5].min(0), Y[:, :5].max(0)
        if a.dim == 6:
            lo, hi = np.append(lo, Y[vm, 5].min()), np.append(hi, Y[vm, 5].max())
        inbox = lambda S: S[((S[:, :5] >= lo[:5]) & (S[:, :5] <= hi[:5])).all(1)]   # noqa: E731
        X = inbox(X)
        print(f"[{name}] {len(X)} simulated stars inside the members' box")
        lc_ = {"stream": cols[j], "stream-dark": darker(cols[j])}.get(a.line_color, a.line_color)
        lines = []                                          # (stars, linestyle, label): same chain + box cut, all stars
        for path, ls, lab in ((a.prior, "-", a.prior_label), (a.overlay, "--", a.overlay_label)):
            if path:
                Z = stars(load_sim_groups(path.format(name=name), a.n_sim, rng)[0], j)
                Z = inbox(Z)
                print(f"[{name}] {lab}: {len(Z)} stars inside the members' box")
                lines.append((Z, ls, lab))
        Xc = X                                              # contours use every box star
        if 0 < a.n_points < len(X):
            X = X[rng.choice(len(X), a.n_points, replace=False)]
        al = a.alpha if a.alpha > 0 else float(np.clip(1e5 / max(len(X), 1), 0.01, 0.35))   # 0 = scale with the star count
        rng_ = list(zip(lo, hi))  # the stars are cut to this box: axes end exactly at it, no padding

        nd = a.dim - 1
        fig, axs = plt.subplots(nd, nd, figsize=(2.75 * nd, 2.75 * nd))
        for r in range(1, a.dim):
            for c in range(nd):
                ax = axs[r - 1, c]
                if c >= r:
                    ax.axis("off"); continue
                if a.contours:
                    xc, yc, H, lv = density_levels(Xc[:, c], Xc[:, r], rng_[c], rng_[r])
                    ax.contourf(xc, yc, H, levels=[lv[0], lv[1], np.inf], colors=[cols[j]], alpha=[0.35, 0.7], zorder=1)
                    if not a.line_color.startswith("stream"):
                        ax.contour(xc, yc, H, levels=lv, colors=[cols[j]], linewidths=1.2, zorder=2)
                else:
                    ax.plot(X[:, c], X[:, r], ".", ms=a.ms, color=cols[j], alpha=al, mew=0, rasterized=True)
                ym = vm if 5 in (c, r) else np.ones(len(Y), bool)   # v_los panels: measured members only
                ax.plot(Y[ym, c], Y[ym, r], "o", ms=2.6, color="k", alpha=0.85, mew=0, zorder=10)
                for Z, ls, _ in lines:
                    xc, yc, H, lv = density_levels(Z[:, c], Z[:, r], rng_[c], rng_[r])
                    ax.contour(xc, yc, H, levels=lv, colors=[lc_], linestyles=ls, linewidths=LINE_WIDTH, zorder=12)
                ax.set_xlim(rng_[c]); ax.set_ylim(rng_[r])
                if r == nd:
                    ax.set_xlabel(LABELS[c])
                    ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))   # no clash with the next panel
                else:
                    ax.set_xticklabels([])
                if c == 0:
                    ax.set_ylabel(LABELS[r])
                else:
                    ax.set_yticklabels([])
        h = [plt.Line2D([], [], ls="", marker="o", color=cols[j]), plt.Line2D([], [], ls="", marker="o", color="k")]
        h0 = [Patch(facecolor=cols[j], edgecolor=cols[j], alpha=0.7)] + h[1:] if a.contours else h
        hc = h0 + [plt.Line2D([], [], ls=ls, lw=LINE_WIDTH, color=lc_) for _, ls, _ in lines]
        lc = [a.label, "Gaia"] + [lab for _, _, lab in lines]
        fig.legend(hc, lc,
                   loc="upper right", bbox_to_anchor=(0.9, 0.86), fontsize=a.fontsize * 1.15 if a.fontsize else 13, frameon=False)
        fig.subplots_adjust(hspace=0.06, wspace=0.06)
        p = os.path.join(a.out, f"ppc_corner_{'contours_' if a.contours else ''}{'6d_' if a.dim == 6 else ''}{name}.pdf")
        fig.savefig(p, dpi=200, bbox_inches="tight"); fig.savefig(p[:-4] + ".png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p)

        # sky-only corner (alpha, delta) WITH the 1-D marginals on the diagonal
        fig, axs = plt.subplots(2, 2, figsize=(6.5, 6.5))
        sky_corner(axs, X, Y, rng_, cols[j], al, a.ms)
        fig.legend(h, [a.label, "Gaia"], loc="upper right", bbox_to_anchor=(0.9, 0.86), fontsize=12, frameon=False)
        fig.subplots_adjust(hspace=0.06, wspace=0.06)
        p = os.path.join(a.out, f"ppc_corner_sky_{name}.pdf")
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        print("wrote", p)
        sky[name] = dict(X=X, Xc=Xc, Y=Y, rng_=rng_, c=cols[j], al=al, h=h, hc=hc, lc=lc, lines=lines, lc_=lc_)

    # the three sky corners side by side (3:1), bigger labels, delta label only on the first;
    # with --contours the stream panels are filled 68/95 % contours of the simulated stars only (no
    # --prior/--overlay lines), and a second figure shows the --prior stars alone as filled contours
    order = [n for n in ("M68", "NGC3201", "Pal5") if n in sky]
    prior_only = a.contours and a.prior and all(any(lab == a.prior_label for _, _, lab in sky[n]["lines"]) for n in order)
    for stem in ["ppc_corner_sky_all"] + (["prior_corner_sky_all"] if prior_only else []):
        fig = plt.figure(figsize=(6.5 * len(order), 6.5))
        for sf, name in zip(fig.subfigures(1, len(order), wspace=0), order):
            d = sky[name]
            first = name == order[0]  # only the first corner carries the delta label -> wider left margin
            axs = sf.subplots(2, 2, gridspec_kw={"hspace": 0.06, "wspace": 0.06, "left": 0.14 if first else 0.11,
                                                 "right": 0.99})
            if not a.contours:
                sky_corner(axs, d["X"], d["Y"], d["rng_"], d["c"], d["al"], a.ms, fs=22, ylabel=first)
                hh, ll = d["h"], [a.label, "Gaia"]
            elif stem.startswith("prior"):
                Z = next(Z for Z, _, lab in d["lines"] if lab == a.prior_label)
                sky_corner(axs, None, d["Y"], d["rng_"], d["c"], 0, a.ms, fs=22, ylabel=first, Xc=Z)
                hh, ll = d["hc"][:2], [a.prior_label, "Gaia"]
            else:
                sky_corner(axs, None, d["Y"], d["rng_"], d["c"], 0, a.ms, fs=22, ylabel=first, Xc=d["Xc"])
                hh, ll = d["hc"][:2], [a.label, "Gaia"]
            axs[0, 1].legend(hh, ll, loc="center", fontsize=18, frameon=False, title=name, title_fontsize=18)
        p = os.path.join(a.out, stem + ".pdf")
        fig.savefig(p, dpi=200, bbox_inches="tight"); fig.savefig(p[:-4] + ".png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p)


def darker(rgba, factor=0.6):
    """Same hue and saturation, lightness scaled by `factor` (HLS)."""
    import colorsys
    h, l, s = colorsys.rgb_to_hls(*rgba[:3])
    return colorsys.hls_to_rgb(h, l * factor, s)


def sky_corner(axs, X, Y, rng_, color, al, ms, fs=None, ylabel=True, Xc=None, lines=(), line_color=None):
    """(alpha, delta) scatter + 1-D marginals of simulated stars X vs members Y on a 2x2 grid. With Xc the
    stars are drawn as filled 68/95 % contours instead, and each (stars, linestyle, label) of `lines` as
    contour lines + step marginals in line_color."""
    S = X if Xc is None else Xc
    for k in range(2):
        ax = axs[k, k]
        ax.hist(S[:, k], bins=60, range=rng_[k], density=True, histtype="stepfilled", color=color, alpha=0.6, lw=0)
        for Z, ls, _ in lines:
            ax.hist(Z[:, k], bins=60, range=rng_[k], density=True, histtype="step", color=line_color, ls=ls, lw=LINE_WIDTH)
        ax.hist(Y[:, k], bins=30, range=rng_[k], density=True, histtype="step", color="k", lw=1.4)
        ax.set_xlim(rng_[k]); ax.set_yticks([])
    ax = axs[1, 0]
    if Xc is None:
        ax.plot(X[:, 0], X[:, 1], ".", ms=ms, color=color, alpha=al, mew=0, rasterized=True)
    else:
        xc, yc, H, lv = density_levels(Xc[:, 0], Xc[:, 1], rng_[0], rng_[1])
        ax.contourf(xc, yc, H, levels=[lv[0], lv[1], np.inf], colors=[color], alpha=[0.35, 0.7], zorder=1)
        for Z, ls, _ in lines:
            xc, yc, H, lv = density_levels(Z[:, 0], Z[:, 1], rng_[0], rng_[1])
            ax.contour(xc, yc, H, levels=lv, colors=[line_color], linestyles=ls, linewidths=LINE_WIDTH, zorder=12)
    ax.plot(Y[:, 0], Y[:, 1], "o", ms=2.6, color="k", alpha=0.85, mew=0, zorder=10)
    ax.set_xlim(rng_[0]); ax.set_ylim(rng_[1])
    ax.set_xlabel(LABELS[0], fontsize=fs)
    if ylabel:
        ax.set_ylabel(LABELS[1], fontsize=fs)
    axs[1, 1].set_xlabel(LABELS[1], fontsize=fs)
    axs[0, 0].set_xticklabels([]); axs[0, 1].axis("off")


if __name__ == "__main__":
    main()
