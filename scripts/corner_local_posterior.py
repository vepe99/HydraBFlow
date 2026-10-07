"""Per-stream corner plots of a LOCAL-level real-data posterior (evaluate composition=local, ancestral).

Same look as ``replot_global_corner.py --shaded --kde --prior-bounds`` (bright fill + line inside
68 %, faint fill to 95 %, median dot, reflected-KDE diagonals, grey dashed uniform-prior bounds),
one figure per stream in its RdYlBu_r stream colour. Normal priors are drawn as a grey dashed
density on the diagonal (they have no bounds).

    python scripts/corner_local_posterior.py RUN_DIR/eval_real [--out-prefix local_corner]
"""

import argparse
import os
import sys

import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from replot_global_corner import STREAMS, kde_1d  # noqa: E402

# (key, label, scale)
PARAMS = [
    ("m_progenitor", r"$M_{\rm prog}\ [10^4\,M_\odot]$", 1e-4),
    ("t_end", r"$t_{\rm end}\ [{\rm Gyr}]$", 1.0),
    ("vr", r"$v_r\ [{\rm km\,s^{-1}}]$", 1.0),
    ("r", r"$d\ [{\rm kpc}]$", 1.0),
    ("mu_ra_cosdec", r"$\mu_{\alpha*}\ [{\rm mas\,yr^{-1}}]$", 1.0),
    ("mu_dec", r"$\mu_\delta\ [{\rm mas\,yr^{-1}}]$", 1.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--out-prefix", default="local_corner")
    args = ap.parse_args()
    post = np.load(os.path.join(args.eval_dir, "posterior.npz"))
    pri = yaml.safe_load(open(os.path.join(args.eval_dir, ".hydra/config.yaml")))["simulator"]["params"]["priors_local"]
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 1 + len(STREAMS)))[1:]
    K = len(PARAMS)

    for j, (stream, c) in enumerate(zip(STREAMS, colors)):
        data = np.column_stack([np.asarray(post[k])[0, j].reshape(-1) * s for k, _, s in PARAMS])
        priors = [(pri[stream][k]["type"], np.array(pri[stream][k]["prior_parameters"]) * s) for k, _, s in PARAMS]
        lo, hi = np.nanpercentile(data, [0.5, 99.5], axis=0)
        ranges = []
        for i, (t, p) in enumerate(priors):  # widen to show the prior: bounds, or +-3 sigma for normals
            a, b = (p[0], p[1]) if t == "uniform" else (p[0] - 3 * p[1], p[0] + 3 * p[1])
            a, b = min(lo[i], a), max(hi[i], b)
            ranges.append((a - 0.05 * (b - a), b + 0.05 * (b - a)))

        fig = corner.corner(
            data, labels=[p[1] for p in PARAMS], range=ranges, color=c, bins=40, smooth=1.0,
            levels=(0.68, 0.95), plot_datapoints=False, plot_density=False,
            fill_contours=True, no_fill_contours=True, label_kwargs={"fontsize": 16},
            hist_kwargs={"density": True, "alpha": 0},
            contourf_kwargs={"colors": [(*c[:3], 0.0), (*c[:3], 0.05), (*c[:3], 0.35)]},
            contour_kwargs={"colors": [(*c[:3], 0.0), (*c[:3], 1.0)]},  # no 2-sigma line
        )
        corner.overplot_points(fig, [np.nanmedian(data, axis=0)], marker="o", color=c, ms=6, mec="k", mew=0.5)
        axes = np.array(fig.axes).reshape(K, K)
        line = dict(color="0.4", ls="--", lw=1.0, zorder=0)
        for i, (t, p) in enumerate(priors):
            ax, x = axes[i, i], np.linspace(*ranges[i], 400)
            inside = t == "uniform" and p[0] <= np.nanmin(data[:, i]) and np.nanmax(data[:, i]) <= p[1]
            y = kde_1d(data[:, i], x, tuple(p) if inside else None)  # no reflection if it leaks past a bound
            ax.plot(x, y, color=c, lw=2.0)
            top = y.max()
            if t == "normal":
                yp = norm.pdf(x, *p)
                ax.plot(x, yp, **line)
                top = max(top, yp.max())
            ax.set_ylim(0, 1.1 * top)
            if t == "uniform":
                for v in p:
                    for r in range(i, K):
                        axes[r, i].axvline(v, **line)
                    for col in range(i):
                        axes[i, col].axhline(v, **line)
        handles = [Line2D([0], [0], color=c, lw=2, label=f"{stream} (local posterior)"),
                   Line2D([0], [0], color="0.4", ls="--", label="Prior (bounds / normal density)")]
        fig.legend(handles=handles, loc="upper right", fontsize=14, frameon=False)
        path = os.path.join(args.eval_dir, f"{args.out_prefix}_{stream}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
