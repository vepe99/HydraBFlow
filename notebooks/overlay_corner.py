"""One corner plot per model: the training prior shaded behind, every condition regime overlaid.

Eight posteriors from three evaluate runs of the same weights on Oph 163131:

  regime            colour      curves
  both supplied     blue        the four (sil_id, has_cavity) branches
  sil_id masked     vermillion  two, one per has_cavity  (the posterior no longer depends on sil_id)
  has_cavity masked green       two, one per sil_id

Colour encodes what the network was told; linestyle separates the branches inside a regime.
1-sigma contours only, unfilled -- eight filled sets would be mud.

Usage: overlay_corner.py <real_oph_dir> <out.png> <param> [<param> ...]
"""
import sys

import corner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Okabe-Ito, colourblind-safe.
COND, SIL, CAV = "#0072B2", "#D55E00", "#009E73"
PRIOR_GREY = "#d8d8d8"
LS = ["-", "--", "-.", ":"]
# (row index in posterior.npz, label) per regime. Row order is itertools.product over
# [sil_id, has_cavity]: (0,0), (0,1), (1,0), (1,1).
REGIMES = [
    ("mask_none", COND, [(0, "sil 0, cav 0"), (1, "sil 0, cav 1"),
                         (2, "sil 1, cav 0"), (3, "sil 1, cav 1")], "both supplied"),
    # sil_id unobserved: rows 0/2 and 1/3 are identical, so one per has_cavity is enough.
    ("mask_sil_id", SIL, [(0, "cav 0"), (1, "cav 1")], "sil_id masked"),
    # has_cavity unobserved: rows 0/1 and 2/3 are identical.
    ("mask_has_cavity", CAV, [(0, "sil 0"), (2, "sil 1")], "has_cavity masked"),
]


def chains(run_dir, params):
    z = np.load(f"{run_dir}/posterior.npz")
    return np.stack([np.asarray(z[p]).reshape(z[p].shape[0], -1) for p in params], axis=-1)


def main(root, out, params):
    zb = np.load(f"{root}/prior_bounds.npz")
    bounds = [tuple(float(x) for x in zb[p]) if p in zb else None for p in params]
    pad = [None if b is None else (b[0] - .05 * (b[1] - b[0]), b[1] + .05 * (b[1] - b[0]))
           for b in bounds]
    ranges = [1.0 if r is None else r for r in pad]

    fig, handles = None, []
    for sub, colour, rows, regime in REGIMES:
        c = chains(f"{root}/{sub}", params)
        for k, (row, label) in enumerate(rows):
            fig = corner.corner(
                c[row], labels=params, range=ranges, fig=fig, color=colour,
                plot_datapoints=False, plot_density=False, fill_contours=False,
                levels=(0.393,), smooth=1.0, hist_kwargs=dict(lw=1.2, ls=LS[k]),
                contour_kwargs=dict(linewidths=1.1, linestyles=LS[k]),
                labelpad=0.08,
            )
            handles.append(Line2D([], [], color=colour, ls=LS[k], lw=1.4,
                                  label=f"{regime} — {label}"))

    # Prior box behind everything.
    n = len(params)
    axes = np.array(fig.axes).reshape(n, n)
    for r in range(n):
        for col in range(r + 1):
            ax = axes[r, col]
            if bounds[col]:
                ax.axvspan(*bounds[col], color=PRIOR_GREY, zorder=-10, lw=0)
            if r != col and bounds[r]:
                ax.axhspan(*bounds[r], color=PRIOR_GREY, zorder=-10, lw=0)
    handles.insert(0, Patch(facecolor=PRIOR_GREY, edgecolor="none",
                            label="training prior (min–max)"))

    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=13 if n > 8 else 10,
               bbox_to_anchor=(0.97, 0.97), title="Oph 163131 · 1σ contours",
               title_fontsize=15 if n > 8 else 11)
    fig.savefig(out, bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  ({n} params, {len(handles) - 1} posteriors)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
