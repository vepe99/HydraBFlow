"""Paper corner plots of the per-stream LOCAL posterior of a composition=local real-data run
(<run>/posterior.npz, physical units), one figure per stream in its stream colour, with the prior
from the run's simulator config as dashed lines: uniform priors as their bounds (vertical lines on
the diagonal and in every panel of that column, horizontal in every panel of that row), normal
priors as their Gaussian density on the diagonal. Writes <run>/paper_plots/local_corner_<stream>.

    python scripts/plot_local_corner.py EVAL_REAL_DIR [FONT_SCALE]
"""

import os
import sys

import corner
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import norm

from plot_calibration_ecdf import LABEL, LOCAL_KEYS, STREAM_COLOR, STREAMS, save

UNIT = {"m_progenitor": r"[$10^{4}\,\mathrm{M_\odot}$]", "t_end": "[Gyr]", "vr": r"[$\mathrm{km\,s^{-1}}$]",
        "r": "[kpc]", "mu_ra_cosdec": r"[$\mathrm{mas\,yr^{-1}}$]", "mu_dec": r"[$\mathrm{mas\,yr^{-1}}$]"}
SCALE = {"m_progenitor": 1e4}
LEVELS = (0.68, 0.95)


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


def main(run, fs_scale=1.0):
    cfg = yaml.safe_load(open(os.path.join(run, ".hydra", "config.yaml")))
    priors_local = cfg["simulator"]["params"]["priors_local"]
    post = np.load(os.path.join(run, "posterior.npz"))
    out = os.path.join(run, "paper_plots")
    os.makedirs(out, exist_ok=True)
    fs = 13 * fs_scale
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs * 1.15, "xtick.labelsize": fs * 0.85,
                         "ytick.labelsize": fs * 0.85})
    keys = [k for k in LOCAL_KEYS if k in post]
    labels = [f"{LABEL[k]} {UNIT[k]}" for k in keys]
    for j, name in enumerate(STREAMS):
        col = STREAM_COLOR[name]
        X = np.column_stack([post[k][0, j, :, 0] / SCALE.get(k, 1) for k in keys])
        pri = [Prior(priors_local[name][k], SCALE.get(k, 1)) for k in keys]
        rng = [axis_range(X[:, i], p) for i, p in enumerate(pri)]
        fig = corner.corner(X, labels=labels, range=rng, color=col, levels=LEVELS, bins=30, smooth=1.0,
                            fill_contours=True, plot_datapoints=False, plot_density=False,
                            hist_kwargs={"density": True, "lw": 2},
                            contour_kwargs={"linewidths": 1.5}, max_n_ticks=3, labelpad=0.08)
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
        fig.legend(h, ["Posterior", "Prior"], loc="upper right", bbox_to_anchor=(0.92, 0.9), frameon=False,
                   fontsize=fs * 1.3, title=name, title_fontsize=fs * 1.5)
        save(fig, out, f"local_corner_{name}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
