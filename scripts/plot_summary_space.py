"""Learned summary space, simulated test set vs the real Gaia members (companion to mmd_hypothesis_test.png).

The fused summaries of ``evaluate`` (``summaries.npz``: one row per member, ``j`` = stream index) are the
concatenation [stream backbone | rotation-curve backbone] (adapter ``summary_variables`` order). The masked
compositional plan conditions on items with the unobserved block zeroed: [stream_j, 0] per member and
[0, curve] once, so that is the common space here. The simulated items set the principal components (linear,
so distances keep their meaning; the real items are only projected). The curve is group-level: the real
members share one curve item, the simulated ones differ by the per-member observational noise.

    pca_scatter    PC1 vs PC2: stream items coloured by stream, curve items purple; Gaia = stars
    pca_corner     first --n-pc components, per-group 68/95 % contours

    python scripts/plot_summary_space.py --ref <eval_sim>/summaries.npz --obs <eval_real>/summaries.npz \
        --stream-dim 44 --out <dir>
"""

import argparse
import json
import os

import corner
import matplotlib
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STREAMS = ["Pal5", "NGC3201", "M68"]  # j order
CURVE = "#984ea3"  # rotation-curve purple of the corner plots


def pca(X, Y):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs, Ys = (X - mu) / sd, (Y - mu) / sd
    _, s, vt = np.linalg.svd(Xs, full_matrices=False)
    return Xs @ vt.T, Ys @ vt.T, s**2 / np.sum(s**2)


def pc_labels(ev, k):
    return [f"PC{i + 1} ({100 * ev[i]:.0f} %)" for i in range(k)]


def draw_corner(P, Q, groups, labels, path, fs, handles):
    """groups: (colour, simulated rows, real row) per cloud."""
    k = len(labels)
    lo = np.minimum(np.percentile(P[:, :k], 0.5, axis=0), Q[:, :k].min(0))
    hi = np.maximum(np.percentile(P[:, :k], 99.5, axis=0), Q[:, :k].max(0))
    pad = 0.05 * (hi - lo)
    rng = list(zip(lo - pad, hi + pad))
    fig = None
    for c, m, _ in groups:
        fig = corner.corner(P[m, :k], fig=fig, range=rng, color=c, labels=labels, bins=30, smooth=1.0,
                            levels=(0.68, 0.95), plot_datapoints=False, plot_density=False, fill_contours=True,
                            hist_kwargs={"density": True, "alpha": 0.0},  # diagonal redrawn as a KDE below
                            contourf_kwargs={"colors": [to_rgba(c, a) for a in (0.0, 0.15, 0.4)]},
                            contour_kwargs={"colors": [c], "linewidths": 1.5},
                            label_kwargs={"fontsize": fs}, max_n_ticks=3)
    axes = np.array(fig.axes).reshape(k, k)
    for r in range(k):
        x = np.linspace(*rng[r], 400)
        top = 0.0
        for c, m, _ in groups:
            y = gaussian_kde(P[m, r])(x)
            axes[r, r].plot(x, y, color=c, lw=3.0)
            top = max(top, y.max())
        axes[r, r].set_ylim(0, 1.1 * top)
    for c, _, q in groups:
        for r in range(k):
            axes[r, r].axvline(q[r], color=c, lw=2.5, ls="--")
            for col in range(r):
                axes[r, col].scatter(q[col], q[r], color=c, marker="*", s=350, edgecolor="k", lw=1.5, zorder=10)
    for ax in fig.axes:
        ax.tick_params(labelsize=0.7 * fs)
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=fs)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="simulated summaries.npz (evaluate on the test set)")
    ap.add_argument("--obs", required=True, help="real-data summaries.npz (evaluate on Gaia)")
    ap.add_argument("--stream-dim", type=int, required=True,
                    help="width of the stream backbone (first block of the fused summary); the rest is the curve")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pc", type=int, default=5)
    ap.add_argument("--fontsize", type=float, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.size": args.fontsize})
    k, d = args.n_pc, args.stream_dim

    ref, obs = np.load(args.ref), np.load(args.obs)
    X, jx = ref["summaries"].astype(float), ref["j"].astype(int)
    Y, jy = obs["summaries"].astype(float), obs["j"].astype(int)

    def items(Z):  # the compositional plan's items: [stream_j, 0] per member and [0, curve]
        S, C = Z.copy(), Z.copy()
        S[:, d:], C[:, :d] = 0.0, 0.0
        return S, C

    (XS, XC), (YS, YC) = items(X), items(Y)
    P, Q, ev = pca(np.vstack([XS, XC]), np.vstack([YS, YC[:1]]))  # real curve rows identical: one item
    n, m = len(X), len(Y)
    gx = np.r_[jx, np.full(n, 3)]          # group of each simulated item: stream j, or 3 = curve
    gy = np.r_[jy, 3]
    colors = {**dict(zip(range(3), plt.cm.RdYlBu_r(np.linspace(0, 1, 4))[1:])), 3: CURVE}
    names = STREAMS + ["Circular velocity"]
    star = dict(marker="*", s=600, edgecolor="k", linewidth=1.5, zorder=10)
    handles = [Line2D([0], [0], marker="o", ls="", color=colors[g], ms=10, label=names[g]) for g in range(4)]
    handles.append(Line2D([0], [0], marker="*", ls="", mfc="w", mec="k", ms=20, label="Gaia"))
    labels = pc_labels(ev, k)

    # --- PC1 vs PC2
    fig, ax = plt.subplots(figsize=(8, 8))
    for g in range(4):
        ax.scatter(P[gx == g, 0], P[gx == g, 1], s=12, color=colors[g], alpha=0.4, lw=0)
    for g in range(4):
        ax.scatter(Q[gy == g, 0], Q[gy == g, 1], color=colors[g], **star)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_box_aspect(1)
    ax.legend(handles=handles, frameon=False, fontsize=0.8 * args.fontsize)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"pca_scatter.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    draw_corner(P, Q, [(colors[g], gx == g, Q[gy == g][0]) for g in range(4)], labels,
                os.path.join(args.out, "pca_corner"), args.fontsize, handles)

    # percentile of each real item within its own simulated cloud, per PC
    pct = {names[g]: [float(np.mean(P[gx == g, i] < Q[gy == g, i][0]) * 100) for i in range(k)] for g in range(4)}
    summary = {"explained_variance": ev[:k].tolist(), "percentile_per_pc": pct, "n_sim": n, "n_real": m}
    with open(os.path.join(args.out, "summary_space.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
