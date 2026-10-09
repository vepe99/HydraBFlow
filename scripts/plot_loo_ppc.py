"""Paper figure: leave-one-stream-out posterior-predictive check.

Row = held-out stream, column = observable along phi1 (published STREAMFINDER frame). In each panel:
the prediction for that stream from the OTHER two streams + the rotation curve (coloured 5-95 % band +
median), the same with all three streams (dashed), and the Gaia members (dots) with their B-spline
track (black). Inputs are the ppc_joint_curves_aug.npz written by ppc_joint_posterior.py into
<loo>/without_<stream>/ppc_<mode>/ and <loo>/full_baseline/ppc_<mode>/ (run_loo_ppc.sh).

Usage: python scripts/plot_loo_ppc.py <loo dir> [--mode strict|lenient]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

STREAMS = ["Pal5", "NGC3201", "M68"]
OBS = [("phi2", r"$\phi_2$ [deg]", 1), ("parallax", r"$\varpi$ [mas]", 2), ("mu_phi1", r"$\mu_{\phi_1}$ [mas yr$^{-1}$]", 3),
       ("mu_phi2", r"$\mu_{\phi_2}$ [mas yr$^{-1}$]", 4), ("vlos", r"$v_{\rm los}$ [km s$^{-1}$]", 5)]
COLOR = dict(zip(STREAMS, plt.cm.RdYlBu_r(np.linspace(0, 1, 4))[1:]))   # = real_global_vs_streams_corner


def density_levels(x, y, xr, yr, levels=(0.68, 0.95), bins=60, smooth=1.5):
    """Smoothed 2-D histogram and the density thresholds enclosing `levels` of the mass (outer first)."""
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    H = gaussian_filter(H, smooth).T
    flat = np.sort(H.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    lv = sorted(flat[np.searchsorted(cum, lv)] for lv in levels)
    return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H, lv


def inside(c, key):
    ok = c[key + "keep"].astype(bool)
    r = c[key + "real"]
    return float(np.mean(((r >= c[key + "q5"]) & (r <= c[key + "q95"]))[ok])) if ok.any() else np.nan


CORNER_LABELS = [r"$\phi_1$ [deg]"] + [lab for _, lab, _ in OBS]


def corners(args, out):
    """One corner per held-out stream: LOO particles + filled 68/95 % contours in the stream colour,
    the full-data PPC as dashed contours, Gaia members in black. 5-D over all stars, 6-D (+v_los) over
    stars with a measured v_los on both sides."""
    import corner

    for name in STREAMS:
        loo = np.load(os.path.join(args.loo, f"without_{name}", f"ppc_{args.mode}{args.suffix}", "ppc_joint_curves_aug.npz"))
        full = np.load(os.path.join(args.loo, "full_baseline", f"ppc_{args.mode}{args.suffix}", "ppc_joint_curves_aug.npz"))
        X, F, Y = loo[f"{name}__sims"], full[f"{name}__sims"], loo[f"{name}__members"]
        for dim in (5, 6):
            x, f, y = X[:, :dim], F[:, :dim], Y[:, :dim]
            x, f, y = (a[np.isfinite(a).all(axis=1)] for a in (x, f, y))
            if args.limits == "members":
                lo, hi, pad = y.min(axis=0), y.max(axis=0), 0.03
            else:
                lo = np.minimum(np.percentile(x, 0.5, axis=0), y.min(axis=0))   # every member stays in frame
                hi, pad = np.maximum(np.percentile(x, 99.5, axis=0), y.max(axis=0)), 0.08
            rng_ = [(a - pad * (b - a), b + pad * (b - a)) for a, b in zip(lo, hi)]
            kw = dict(labels=CORNER_LABELS[:dim], range=rng_, bins=40, smooth=1.0, levels=(0.68, 0.95))
            fig = corner.corner(x, color=COLOR[name], plot_datapoints=True, fill_contours=True, plot_density=False,
                                data_kwargs=dict(alpha=0.08, ms=1.5), hist_kwargs=dict(density=True, lw=1.4), **kw)
            corner.corner(f, fig=fig, color="0.25", plot_datapoints=False, plot_density=False, no_fill_contours=True,
                          contour_kwargs=dict(linestyles="--", linewidths=0.9),
                          hist_kwargs=dict(density=True, lw=1.0, ls="--"), **kw)
            axs = np.array(fig.axes).reshape(dim, dim)
            for r in range(dim):
                axs[r, r].hist(y[:, r], bins=40, range=rng_[r], density=True, histtype="step", color="k", lw=1.2)
                for c in range(r):
                    axs[r, c].plot(y[:, c], y[:, r], "o", ms=2.2, color="k", alpha=0.8, mew=0, zorder=10)
            handles = [Patch(color=COLOR[name], alpha=0.7, label="held out: predicted from the other two streams + $v_c$ curve"),
                       Line2D([], [], color="0.25", lw=0.9, ls="--", label="all three streams + $v_c$ curve"),
                       Line2D([], [], marker="o", ls="", ms=3, color="k", label="Gaia members" +
                              (" (measured $v_{\\rm los}$)" if dim == 6 else ""))]
            fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=10, bbox_to_anchor=(0.98, 0.98))
            stem = f"loo_ppc_{args.mode}{args.suffix}_corner{dim}d{'_memberlim' if args.limits == 'members' else ''}_{name}"
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}/{stem}  ({len(x)} LOO stars, {len(f)} full, {len(y)} members)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loo")
    ap.add_argument("--mode", default="strict", choices=("strict", "lenient"))
    ap.add_argument("--suffix", default="", help="read ppc_<mode><suffix>/ (e.g. _n500) and tag the output")
    ap.add_argument("--limits", default="auto", choices=("auto", "members"),
                    help="corner: members = axes span the observed stars' min/max (+3 %%); contours then refer to the mass inside")
    ap.add_argument("--style", default="bands", choices=("bands", "particles", "contours", "corner"),
                    help="particles: held-out prediction as simulated stars instead of the B-spline band")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 9})
    out = os.path.join(args.loo, "paper_plot")
    os.makedirs(out, exist_ok=True)
    if args.style == "corner":
        return corners(args, out)

    fig, axes = plt.subplots(len(STREAMS), len(OBS), figsize=(13, 6.6), squeeze=False)
    for s, name in enumerate(STREAMS):
        loo = np.load(os.path.join(args.loo, f"without_{name}", f"ppc_{args.mode}{args.suffix}", "ppc_joint_curves_aug.npz"))
        full = np.load(os.path.join(args.loo, "full_baseline", f"ppc_{args.mode}{args.suffix}", "ppc_joint_curves_aug.npz"))
        rng = np.random.default_rng(0)
        sims = loo[f"{name}__sims"]
        sims = sims[rng.choice(len(sims), min(len(sims), 6000), replace=False)] if args.style == "particles" else sims
        fsims = full[f"{name}__sims"]
        for c, (key, lab, col) in enumerate(OBS):
            ax, k = axes[s, c], f"{name}__{key}__"
            g = full[k + "grid"]
            if args.style in ("particles", "contours"):
                ok = np.isfinite(sims[:, col])
                stars = loo[k + "stars"]
                if args.style == "particles":
                    ax.scatter(sims[ok, 0], sims[ok, col], s=1.5, color=COLOR[name], alpha=0.25, lw=0, zorder=1)
                else:
                    xr = (np.nanmin(g) - 2, np.nanmax(g) + 2)
                    yr = tuple(np.nanpercentile(np.r_[sims[ok, col], stars[:, 1]], [0.5, 99.5]))
                    yr = (yr[0] - 0.2 * (yr[1] - yr[0]), yr[1] + 0.2 * (yr[1] - yr[0]))
                    X, Y, H, lv = density_levels(sims[ok, 0], sims[ok, col], xr, yr)
                    ax.contourf(X, Y, H, levels=[lv[0], lv[1], H.max()], colors=[COLOR[name]], alpha=[0.3, 0.6], zorder=1)
                    ax.contour(X, Y, H, levels=lv, colors=[COLOR[name]], linewidths=0.9, zorder=2)
                    okf = np.isfinite(fsims[:, col])
                    X, Y, H, lv = density_levels(fsims[okf, 0], fsims[okf, col], xr, yr)
                    ax.contour(X, Y, H, levels=lv, colors="0.3", linewidths=0.8, linestyles="--", zorder=3)
                ax.scatter(stars[:, 0], stars[:, 1], s=4, c="k", alpha=0.8, lw=0, zorder=5)
                lo, hi = np.nanpercentile(np.r_[sims[ok, col], stars[:, 1]], [1, 99])
                ax.set_ylim(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo))
                ax.set_xlim(np.nanmin(g) - 2, np.nanmax(g) + 2)
                ax.set_ylabel(lab)
                if s == len(STREAMS) - 1:
                    ax.set_xlabel(r"$\phi_1$ [deg]")
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(direction="in", length=2.5)
                continue
            stars = loo[k + "stars"]
            ax.scatter(stars[:, 0], stars[:, 1], s=3, c="k", alpha=0.25, lw=0, zorder=1)
            for q in ("q5", "q95"):   # outline, so it stays visible over the held-out band
                ax.plot(g, full[k + q], color="0.3", lw=0.9, ls="--", zorder=4.5)
            g = loo[k + "grid"]
            ax.fill_between(g, loo[k + "q5"], loo[k + "q95"], color=COLOR[name], alpha=0.6, lw=0, zorder=3)
            ax.plot(g, loo[k + "q50"], color=COLOR[name], lw=1.3, zorder=4)
            ax.plot(g, loo[k + "real"], color="k", lw=1.6, zorder=5)
            ax.text(0.97, 0.05, f"{inside(loo, k):.2f} / {inside(full, k):.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5, color="0.25")
            band = np.r_[loo[k + "q5"], loo[k + "q95"], full[k + "q5"], full[k + "q95"], loo[k + "real"]]
            lo, hi = np.nanpercentile(band, [1, 99])
            ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
            ax.set_xlim(np.nanmin(g) - 2, np.nanmax(g) + 2)
            ax.set_ylabel(lab)
            if s == len(STREAMS) - 1:
                ax.set_xlabel(r"$\phi_1$ [deg]")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(direction="in", length=2.5)
    handles = [Patch(color="0.45", alpha=0.6, label="held out: predicted from the other two streams + $v_c$ curve (5-95 %)"),
               Line2D([], [], color="0.3", lw=0.9, ls="--", label="all three streams + $v_c$ curve (5-95 %)"),
               Line2D([], [], color="k", lw=1.6, label="Gaia members, B-spline track"),
               Line2D([], [], marker="o", ls="", ms=3, color="k", alpha=0.4, label="Gaia members")]
    if args.style == "contours":
        handles = [Patch(color="0.45", alpha=0.6, label="held out: predicted from the other two streams + $v_c$ curve (68/95 %)"),
                   Line2D([], [], color="0.3", lw=0.8, ls="--", label="all three streams + $v_c$ curve (68/95 %)"),
                   Line2D([], [], marker="o", ls="", ms=3, color="k", label="Gaia members")]
    elif args.style == "particles":
        handles = [Line2D([], [], marker="o", ls="", ms=4, color="0.5", label="held out: simulated stars predicted from the other two streams + $v_c$ curve"),
                   Line2D([], [], marker="o", ls="", ms=3, color="k", label="Gaia members")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"loo_ppc_{args.mode}{args.suffix}{'' if args.style == 'bands' else '_' + args.style}.{ext}"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}/loo_ppc_{args.mode}* ({args.style})")


if __name__ == "__main__":
    main()
