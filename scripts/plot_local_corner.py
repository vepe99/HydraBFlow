"""Paper corner plots of the per-stream LOCAL posterior of a composition=local real-data run
(<run>/posterior.npz, physical units), one figure per stream in its stream colour, with the prior
from the run's simulator config as dashed lines: uniform priors as their bounds (vertical lines on
the diagonal and in every panel of that column, horizontal in every panel of that row), normal
priors as their Gaussian density on the diagonal. Writes <run>/paper_plots/local_corner_<stream>.
With --mmd DIR the stream's bf-default MMD test (DIR/mmd_hypothesis_test_bfdefault_<stream>_<slice>.npz,
written by mmd_local_per_stream.py) is redrawn as an inset in the empty upper right, and the figures
are written as local_corner_<stream>_mmd instead.

    python scripts/plot_local_corner.py EVAL_REAL_DIR [FONT_SCALE] [--mmd DIR] [--slice fused]
"""

import argparse
import os

import corner
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from scipy.stats import norm

from plot_calibration_ecdf import LABEL, LOCAL_KEYS, STREAM_COLOR, STREAMS, save

UNIT = {"m_progenitor": r"[$10^{4}\,\mathrm{M_\odot}$]", "t_end": "[Gyr]", "vr": r"[$\mathrm{km\,s^{-1}}$]",
        "r": "[kpc]", "mu_ra_cosdec": r"[$\mathrm{mas\,yr^{-1}}$]", "mu_dec": r"[$\mathrm{mas\,yr^{-1}}$]"}
SCALE = {"m_progenitor": 1e4}
LEVELS = (0.68, 0.95)
REJECT = "#7f7f7f"  # bf uses orange, too close to the NGC3201 colour


def axis_range(x, prior):
    """Posterior 0.5-99.5 % joined with the prior support (uniform) or mean +- 3 sigma (normal), padded 5 %."""
    lo, hi = np.percentile(x, [0.5, 99.5])
    a, b = prior
    lo, hi = (min(lo, a), max(hi, b)) if prior.kind == "uniform" else (min(lo, a - 3 * b), max(hi, a + 3 * b))
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


class Prior(tuple):
    def __new__(cls, spec, scale):
        a, b = spec["prior_parameters"][:2]
        p = super().__new__(cls, (a / scale, b / scale))
        p.kind = spec["type"]
        return p


def mmd_inset(fig, axs, path, fs, lfs):
    """bf's mmd_hypothesis_test (KDE of the null, 5 % rejection area, observed line) on an inset axis
    spanning the empty corner above the diagonal: columns 3.3-6, rows 0-1.7 from the top."""
    d = np.load(path)
    null, obs = d["mmd_null"], float(d["mmd_observed"])
    n = axs.shape[0]
    a, b, c = axs[0, 3].get_position(), axs[0, n - 1].get_position(), axs[1, 0].get_position()
    ax = fig.add_axes([a.x0 + 0.3 * a.width, c.y0 + 0.3 * c.height, b.x1 - a.x0 - 0.3 * a.width,
                       a.y1 - c.y0 - 0.3 * c.height])
    kde = sns.kdeplot(null, ax=ax, linewidth=0, bw_adjust=1.5)
    x, y = kde.lines[0].get_data()
    ax.fill_between(x, y, color="#132a70", alpha=0.12, lw=0)
    crit = np.quantile(null, 0.95)
    ax.fill_between(x, y, where=x >= crit, interpolate=True, color=REJECT, alpha=0.5, lw=0, label="5% rejection area")
    top = 1.05 * y.max()  # = the y limit, so the lines reach the top of the frame
    ax.vlines(crit, 0, top, color=REJECT, lw=3)
    ax.vlines(obs, 0, top, color="red", lw=3, label="Observed data")
    ax.plot(x, y, color="#132a70", lw=3, label=r"$H_0$")
    ax.set_ylim(0, top)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("MMD", fontsize=fs * 1.15)
    ax.tick_params(labelsize=fs * 0.85)
    hs, ls = ax.get_legend_handles_labels()
    o = [1, 0, 2]  # bf's order: observed, rejection area, H0
    ax.legend([hs[i] for i in o], [ls[i] for i in o], loc="upper right", fontsize=lfs * 1.3)  # mpl default box (white, alpha 0.8), as bf
    return ax


def main(run, fs_scale=1.0, mmd=None, mmd_slice="fused"):
    cfg = yaml.safe_load(open(os.path.join(run, ".hydra", "config.yaml")))
    priors_local = cfg["simulator"]["params"]["priors_local"]
    post = np.load(os.path.join(run, "posterior.npz"))
    out = os.path.join(run, "paper_plots")
    os.makedirs(out, exist_ok=True)
    fs = 13 * fs_scale
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs * 1.15, "xtick.labelsize": fs * 0.85,
                         "ytick.labelsize": fs * 0.85})
    keys = [k for k in LOCAL_KEYS if k in post]
    lfs = 13  # legends keep their size; FONT_SCALE scales only ticks and axis labels
    sep = "\n" if fs_scale > 1 else " "  # scaled labels are wider than a panel: units on a second line
    labels = [f"{LABEL[k]}{sep}{UNIT[k]}" for k in keys]
    for j, name in enumerate(STREAMS):
        col = STREAM_COLOR[name]
        X = np.column_stack([post[k][0, j, :, 0] / SCALE.get(k, 1) for k in keys])
        pri = [Prior(priors_local[name][k], SCALE.get(k, 1)) for k in keys]
        rng = [axis_range(X[:, i], p) for i, p in enumerate(pri)]
        fig = corner.corner(X, labels=labels, range=rng, color=col, levels=LEVELS, bins=30, smooth=1.0,
                            fill_contours=True, plot_datapoints=False, plot_density=False,
                            hist_kwargs={"density": True, "lw": 2},
                            contour_kwargs={"linewidths": 1.5}, max_n_ticks=3, labelpad=0.08 * fs_scale)
        n = len(keys)
        axs = np.array(fig.axes).reshape(n, n)
        for i in range(n):                          # neighbouring end ticks would collide
            axs[n - 1, i].xaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
            if i:
                axs[i, 0].yaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
        pkw = {"color": "k", "ls": "--", "lw": 1.6, "zorder": 5}
        for i, p in enumerate(pri):
            if p.kind == "uniform":
                for v in p:
                    for r in range(i, n):           # column i: diagonal + panels below
                        axs[r, i].axvline(v, **pkw)
                    for c in range(i):              # row i: panels left of the diagonal
                        axs[i, c].axhline(v, **pkw)
            elif p.kind == "normal":
                g = np.linspace(*rng[i], 400)
                axs[i, i].plot(g, norm.pdf(g, *p), **pkw)
                axs[i, i].set_ylim(0, max(axs[i, i].get_ylim()[1], 1.05 * norm.pdf(p[0], *p)))
        h = [plt.Line2D([], [], color=col, lw=3), plt.Line2D([], [], **{**pkw, "zorder": None})]
        if mmd is None:
            fig.legend(h, ["Posterior", "Prior"], loc="upper right", bbox_to_anchor=(0.92, 0.9), frameon=False,
                       fontsize=lfs * 1.3, title=name, title_fontsize=lfs * 1.5)
            save(fig, out, f"local_corner_{name}")
            continue
        mmd_inset(fig, axs, os.path.join(mmd, f"mmd_hypothesis_test_bfdefault_{name}_{mmd_slice}.npz"), fs, lfs)
        p = axs[0, 1].get_position()               # the empty square above the t_end diagonal
        fig.legend(h, ["Posterior", "Prior"], loc="center", bbox_to_anchor=((p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2),
                   frameon=False,
                   fontsize=lfs * 1.3, title=name, title_fontsize=lfs * 1.5)
        save(fig, out, f"local_corner_{name}_mmd")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("font_scale", nargs="?", type=float, default=1.0)
    ap.add_argument("--mmd", default=None, help="dir with mmd_hypothesis_test_bfdefault_<stream>_<slice>.npz")
    ap.add_argument("--slice", default="fused", choices=["fused", "sim_summary", "vcirc_kms"])
    a = ap.parse_args()
    main(a.run, a.font_scale, a.mmd, a.slice)
