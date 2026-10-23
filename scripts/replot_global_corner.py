"""Replot real_global_vs_streams_corner.png from a saved evaluate-real run, with LaTeX labels.

Reads posterior.npz (pooled global) + single_stream_posterior.npz (per member, j order) from
RUN_DIR, maps the log10-preprocessed keys back to physical units, and draws the same overlay
as ``evaluate_real._save_global_vs_streams_corner``.

    python scripts/replot_global_corner.py RUN_DIR [--out NAME] [--curve-only DIR] [--stream-only DIR]

--curve-only / --stream-only add the modality-masked posteriors of the same model (evaluate with
eval.observed_groups=[vcirc_kms] / [sim_summary]; their single_stream_posterior.npz): the curve alone
(black, dashed; identical for every member, member 0 used) and each stream without the curve (dotted,
stream colours) -- the pieces the masked compositional posterior pools.
"""

import argparse
import os
import re

import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D  # noqa: E402

# (key, label, stored as log10 in posterior.npz?)
PARAMS = [
    ("gamma_TwoPowerTriaxial_halo", r"$\gamma$", False),
    ("q_TwoPowerTriaxial_halo", r"$q$", False),
    ("log10_M200_TwoPowerTriaxial_halo", r"$\log_{10}(M_{200}^{h}/M_\odot)$", False),
    ("ln_cvprime_TwoPowerTriaxial_halo", r"$\ln c_{V'}$", False),
    ("Sigma_Disk", r"$\Sigma_D\ [M_\odot\,{\rm kpc^{-2}}]$", True),
    ("z_Disk", r"$z_D\ [{\rm kpc}]$", True),
    ("r_Disk", r"$r_D\ [{\rm kpc}]$", True),
]
STREAMS = ["Pal5", "NGC3201", "M68"]  # j order, as in the evaluate run


# --halo-agama: (log10 M200, ln c_v') replaced by the halo (scaleRadius, densityNorm) AGAMA receives
HALO_PARAMS = [("log10_r_h", r"$\log_{10}(r_h/{\rm kpc})$", False),
               ("log10_rho_h", r"$\log_{10}(\rho_h/M_\odot\,{\rm kpc^{-3}})$", False)]
HALO_IN = ("gamma_TwoPowerTriaxial_halo", "q_TwoPowerTriaxial_halo",
           "log10_M200_TwoPowerTriaxial_halo", "ln_cvprime_TwoPowerTriaxial_halo")
_RUN_DIR = None  # set in main: its .hydra config gives the fixed halo constants + cosmology
_W = {}


def _halo_init(run_dir):
    os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", HYDRABFLOW_NUM_GPUS="0")
    from omegaconf import OmegaConf
    from hydrabflow.registry import get_simulator
    from hydrabflow.simulators.stream_agama import _agama, _resolve_pot_cfg
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra/config.yaml"))
    pri = OmegaConf.to_container(cfg.simulator.params.priors_global)
    _W["consts"] = {k: float(v["prior_parameters"][0]) for k, v in pri.items() if v["type"] == "identity"}
    _W["cfg"] = _resolve_pot_cfg(get_simulator(cfg.simulator)._pot_cfg)
    _W["agama"] = _agama()
    _W["agama"].setNumThreads(1)


def _halo_one(vals):
    from hydrabflow.simulators.stream_agama import _halo_params_m200c
    try:
        hp = _halo_params_m200c(_W["agama"], {**_W["consts"], **dict(zip(HALO_IN, map(float, vals)))}, _W["cfg"])
        return np.log10(hp["scaleRadius"]), np.log10(hp["densityNorm"])
    except Exception:  # e.g. gamma >= 2 leaking past the prior makes r_h non-positive
        return np.nan, np.nan


def halo_cols(npz, member):
    """(n_draws, 2) log10 (r_h, rho_h) per draw, exactly as the simulator converts; cached next to the npz."""
    src = npz.zip.filename
    cache = os.path.join(os.path.dirname(src), f"halo_agama_{os.path.basename(src)[:-4]}_m{member}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    from multiprocessing import Pool
    x = np.column_stack([np.asarray(npz[k])[member].reshape(-1) for k in HALO_IN])
    with Pool(int(os.environ.get("HALO_WORKERS", 32)), _halo_init, (_RUN_DIR,)) as pool:
        out = np.array(pool.map(_halo_one, x, chunksize=500))
    np.save(cache, out)
    print(f"{cache}: {np.isfinite(out[:, 0]).sum()}/{len(out)} draws converted")
    return out


def stack(npz, member):
    cols = []
    for key, _, is_log in PARAMS:
        if key in ("log10_r_h", "log10_rho_h", "r_h", "rho_h"):
            x = halo_cols(npz, member)[:, 0 if key.endswith("r_h") else 1]
            cols.append(x if key.startswith("log10") else 10.0**x)
            continue
        if key == "M_D":  # exponential-disc mass 2 pi Sigma_D r_D^2
            sig, rd = (10.0**np.asarray(npz[k])[member].reshape(-1) for k in ("Sigma_Disk", "r_Disk"))
            cols.append(2 * np.pi * sig * rd**2)
            continue
        x = np.asarray(npz[key])[member].reshape(-1)
        cols.append(10.0**x if is_log else x)
    return np.column_stack(cols)


def prior_bounds(run_dir):
    """(lo, hi) of each PARAMS entry with a uniform prior in the run's simulator config, else None."""
    import yaml
    cfg = yaml.safe_load(open(os.path.join(run_dir, ".hydra/config.yaml")))
    pri = cfg["simulator"]["params"]["priors_global"]
    return [tuple(pri[k]["prior_parameters"]) if k in pri and pri[k]["type"] == "uniform" else None
            for k, _, _ in PARAMS]


def kde_1d(x, grid, bound=None):
    """Gaussian KDE of x on grid; reflected at a hard (uniform-prior) bound so the edge is not under-weighted."""
    from scipy.stats import gaussian_kde
    x = x[np.isfinite(x)]
    if bound is None:
        return gaussian_kde(x)(grid)
    lo, hi = bound
    k = gaussian_kde(x)
    y = k(grid) + k(2 * lo - grid) + k(2 * hi - grid)
    return np.where((grid >= lo) & (grid <= hi), y, 0.0)


def rc_inset(fig, axes, cache, chains, fs, r_break=30.0, obs_label="Ou et al. 2024", col=3.45, bottom_row=2.25,
             ts=0.75, top_row=0.0, lfs=None):
    """Rotation-curve PPC (median + 68 % band per posterior) in the empty upper-right triangle, broken R axis:
    linear to r_break, log beyond (as ppc_rotation_curve_ibata.py's overlay), with the observed curves."""
    from hydrabflow.simulators import stream_common as sc

    z = np.load(cache)
    K = axes.shape[0]
    # left edge at fractional column `col`, bottom at fractional row `bottom_row` (grid units, from the top-left)
    lfs = lfs or fs  # axis-label size of the inset
    a, b = axes[int(top_row), int(col)].get_position(), axes[0, K - 1].get_position()
    c = axes[int(bottom_row), 0].get_position()
    x0, x1 = a.x0 + (col % 1) * a.width, b.x1
    y0, y1 = c.y1 - (bottom_row % 1) * c.height, a.y1 - (top_row % 1) * a.height
    xm = x0 + 0.57 * (x1 - x0)
    axl, axr = fig.add_axes([x0, y0, xm - x0, y1 - y0]), fig.add_axes([xm, y0, x1 - xm, y1 - y0])
    key = {"Compositional": "Combined", "Circular velocity": "Curve"}
    r = z["r_dense"]
    for ax in (axl, axr):
        for name, _, col, _ in chains:
            vc = z[f"{key.get(name, name.split(' ')[0])}_dense"]
            lo, hi = np.nanpercentile(vc, [16, 84], axis=0)
            ax.fill_between(r, lo, hi, color=col, alpha=0.25, lw=0)
            ax.plot(r, np.nanmedian(vc, axis=0), color=col, lw=2.5 if name == "Compositional" else 1.5, label=name)
        eb = dict(elinewidth=1.0, capsize=2, lw=0, ms=5)
        ax.errorbar(z["r_obs"], z["vc_obs"], yerr=z["sigma_obs"], fmt="o", color="0.1", ecolor="0.45",
                    label=obs_label, zorder=5, **eb)
        hu = sc.HUANG_R_KPC > r_break
        ax.errorbar(sc.HUANG_R_KPC[hu], sc.HUANG_VC_KMS[hu], yerr=sc.HUANG_SIGMA_VC[hu], fmt="s", mfc="white",
                    color="0.3", ecolor="0.6", label="Huang et al. 2016", zorder=4, **eb)
        ax.tick_params(labelsize=ts * fs)
    axl.set_xlim(0.1, r_break)
    for ax in (axl, axr):  # empty band at the bottom holds the legend
        ax.set_ylim(100, 255)
    axl.set_yticks(range(100, 251, 50))
    axr.set_xscale("log")
    axr.set_xlim(r_break, r.max())
    axr.set_xticks([50, 100])
    # unlabelled minor ticks: evenly spaced on the linear side, log-spaced past the break
    axl.xaxis.set_minor_locator(plt.matplotlib.ticker.MultipleLocator(2))
    axr.xaxis.set_minor_locator(plt.matplotlib.ticker.LogLocator(subs=np.arange(2, 10)))
    axr.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    for ax in (axl, axr):
        ax.tick_params(axis="x", which="major", length=9, width=1.2)
        ax.tick_params(axis="x", which="minor", length=5, width=1.0)
    axr.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    axl.spines["right"].set_visible(False)  # one full rectangle around both halves, open only at the break
    axr.spines["left"].set_visible(False)
    axl.set_zorder(axr.get_zorder() + 1)  # the legend spills into the right half
    axl.patch.set_visible(False)
    axr.tick_params(axis="y", which="both", left=False, labelleft=False)
    axl.set_ylabel(r"$V_c$ [km/s]", fontsize=lfs)
    axl.set_xlabel(r"$R$ [kpc]", fontsize=lfs, x=(x1 - x0) / (2 * (xm - x0)))
    # single-modality posteriors left of the linear|log break, compositional + data right of it
    h, lbl = axl.get_legend_handles_labels()
    right = [i for i, n in enumerate(lbl) if n == "Compositional" or "et al." in n]
    left = [i for i in range(len(lbl)) if i not in right]
    kw = dict(fontsize=ts * fs, frameon=False, borderaxespad=0.2)
    legs = [axl.legend([h[i] for i in left], [lbl[i] for i in left], loc="lower left",
                       bbox_to_anchor=(0.12, 0), **kw),
            axr.legend([h[i] for i in right], [lbl[i] for i in right], loc="lower left",
                       bbox_to_anchor=(-0.12, 0), **kw)]
    for leg in legs:
        for ln in leg.get_lines():
            ln.set_linewidth(4.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="real_global_vs_streams_corner_latex.png")
    ap.add_argument("--curve-only", default=None, help="eval dir run with eval.observed_groups=[vcirc_kms]")
    ap.add_argument("--stream-only", default=None, help="eval dir run with eval.observed_groups=[sim_summary]")
    ap.add_argument("--no-joint", action="store_true", help="drop the per-stream (stream + curve) chains")
    ap.add_argument("--no-global", action="store_true", help="drop the pooled Global chain")
    ap.add_argument("--modalities-style", action="store_true",
                    help="look of corner_modalities.py: all solid, curve-only purple, Global drawn last and thick")
    ap.add_argument("--shaded", action="store_true",
                    help="bright fill + solid line inside 1 sigma, faint fill (no line) to 2 sigma, median dots")
    ap.add_argument("--palette", choices=["rdylbu", "okabe-ito"], default="rdylbu",
                    help="okabe-ito: colour-blind-safe, high-contrast stream/curve colours and a black Global")
    ap.add_argument("--kde", action="store_true",
                    help="1-D panels as Gaussian KDEs (reflected at uniform prior bounds) instead of histograms")
    ap.add_argument("--prior-bounds", action="store_true",
                    help="grey dashed lines at the uniform prior bounds (from the run's simulator config)")
    ap.add_argument("--halo-agama", nargs="?", const="log", choices=["log", "linear"],
                    help="show the halo scaleRadius/densityNorm AGAMA receives (log10 axes, or 'linear') "
                         "instead of log10 M200, ln c_v'")
    ap.add_argument("--no-units", action="store_true", help="drop the physical units from the axis labels")
    ap.add_argument("--fontsize", type=float, default=16, help="axis-label size; ticks and legend scale with it")
    ap.add_argument("--rc-inset", default=None,
                    help="ppc_rotation_curve_curves_*.npz cache (ppc_rotation_curve_ibata.py --single-stream "
                         "--curve-only): draw the rotation-curve PPC in the empty upper-right corner; it carries the legend")
    ap.add_argument("--tick-scale", type=float, default=0.75, help="tick-label size as a fraction of --fontsize")
    ap.add_argument("--inset-col", type=float, default=3.45, help="--rc-inset left edge, in corner columns from the left")
    ap.add_argument("--inset-row", type=float, default=2.25, help="--rc-inset bottom edge, in corner rows from the top")
    ap.add_argument("--inset-top", type=float, default=0.0, help="--rc-inset top edge, in corner rows from the top")
    ap.add_argument("--inset-tick-scale", type=float, default=None, help="inset tick/legend size (default --tick-scale)")
    ap.add_argument("--inset-label-scale", type=float, default=1.0, help="inset axis-label size as a fraction of --fontsize")
    ap.add_argument("--xlabel-offset", type=float, default=0.3,
                    help="distance of the bottom-row x labels below their axes, in axes heights (corner default 0.3)")
    ap.add_argument("--obs-label", default="Ou et al. 2024", help="legend name of the observed curve in the --rc-inset")
    ap.add_argument("--params", default=None,
                    help="comma-separated panel order from the PARAMS/HALO keys plus r_h, rho_h, M_D "
                         "(overrides --halo-agama)")
    args = ap.parse_args()
    global _RUN_DIR
    _RUN_DIR = args.run_dir
    if args.halo_agama:
        i = [k for k, _, _ in PARAMS].index("log10_M200_TwoPowerTriaxial_halo")
        PARAMS[i:i + 2] = HALO_PARAMS if args.halo_agama == "log" else [
            ("r_h", r"$r_h\ [{\rm kpc}]$", False), ("rho_h", r"$\rho_h\ [M_\odot\,{\rm kpc^{-3}}]$", False)]
    if args.params:
        known = {p[0]: p for p in PARAMS + HALO_PARAMS + [
            ("r_h", r"$r_h\ [{\rm kpc}]$", False), ("rho_h", r"$\rho_h\ [M_\odot\,{\rm kpc^{-3}}]$", False),
            ("M_D", r"$M_D\ [M_\odot]$", False)]}
        PARAMS[:] = [known[k] for k in args.params.split(",")]
    if args.no_units:
        PARAMS[:] = [(k, re.sub(r"\\ \[.*\]\$$", "$", lab), lg) for k, lab, lg in PARAMS]
    bounds = prior_bounds(args.run_dir)

    glob = np.load(os.path.join(args.run_dir, "posterior.npz"))
    single = np.load(os.path.join(args.run_dir, "single_stream_posterior.npz"))
    colors = list(plt.cm.RdYlBu_r(np.linspace(0, 1, 1 + len(STREAMS))))
    # (label, samples, colour, linestyle)
    chains = [] if args.no_global else [("Compositional", stack(glob, 0), colors[0], "-")]
    if not args.no_joint:
        chains += [(s, stack(single, i), colors[1 + i], "-") for i, s in enumerate(STREAMS)]
    if args.stream_only:
        so = np.load(os.path.join(args.stream_only, "single_stream_posterior.npz"))
        chains += [(s if args.no_joint else f"{s} (stream only)", stack(so, i), colors[1 + i], ":") for i, s in enumerate(STREAMS)]
    if args.curve_only:
        co = np.load(os.path.join(args.curve_only, "single_stream_posterior.npz"))
        chains.append(("Circular velocity", stack(co, 0), "black", "--"))
    if args.modalities_style:
        chains = [(n, d, "#984ea3" if n == "Circular velocity" else c, "-") for n, d, c, _ in chains]
        if not args.no_global:
            chains = chains[1:] + chains[:1]  # Global drawn on top
    if args.palette == "okabe-ito":  # Okabe & Ito (2008); Global black so the pooled result reads first
        pal = {"Pal5": "#E69F00", "NGC3201": "#0072B2", "M68": "#CC79A7", "Circular": "#009E73", "Compositional": "black"}
        chains = [(n, d, next((v for k, v in pal.items() if n.startswith(k)), c), ls) for n, d, c, ls in chains]
    thick = lambda n, a, b: {} if not args.modalities_style else {a: b if n == "Compositional" else 1.0}  # noqa: E731

    all_data = np.vstack([c[1] for c in chains])
    lo, hi = np.nanpercentile(all_data, [0.5, 99.5], axis=0)
    pad = 0.05 * np.where(hi > lo, hi - lo, 1.0)
    ranges = list(zip(lo - pad, hi + pad))
    if args.prior_bounds:  # widen the panels so the bounds are visible
        ranges = [(min(r[0], b[0]), max(r[1], b[1])) if b else r for r, b in zip(ranges, bounds)]

    def shade(c):  # corner's levels are sorted by density: [2 sigma (0.95), 1 sigma (0.68)]
        if not args.shaded:
            return {}
        rgba = lambda a: to_rgba(c, a)  # noqa: E731
        return {"fill_contours": True, "no_fill_contours": True,
                "contourf_kwargs": {"colors": [rgba(0.0), rgba(0.05), rgba(0.35)]},
                "contour_kwargs": {"colors": [rgba(0.0), rgba(1.0)]}}  # no 2-sigma line

    fig = None
    for name, data, c, ls in chains:
        kw = dict(
            plot_datapoints=False, plot_density=False, fill_contours=False,
            hist_kwargs={"density": True, "linestyle": ls, **thick(name, "lw", 2.0), **({"alpha": 0} if args.kde else {})},
            contour_kwargs={"linestyles": ls, **thick(name, "linewidths", 2.0)},
        )
        sh = shade(c)
        kw["contour_kwargs"].update(sh.pop("contour_kwargs", {}))
        kw.update(sh)
        fig = corner.corner(
            data, fig=fig, labels=[p[1] for p in PARAMS], range=ranges,
            color=c, bins=40, max_n_ticks=4, label_kwargs={"fontsize": args.fontsize},
            smooth=1.0, levels=(0.68, 0.95), **kw,
        )
        if args.shaded:
            corner.overplot_points(fig, [np.nanmedian(data, axis=0)], marker="o", color=c, ms=6, mec="k", mew=0.5)
    K = len(PARAMS)
    axes = np.array(fig.axes).reshape(K, K)
    if args.kde:
        for i, (lo_i, hi_i) in enumerate(ranges):
            ax, x, top = axes[i, i], np.linspace(lo_i, hi_i, 400), 0.0
            for name, data, c, ls in chains:
                y = kde_1d(data[:, i], x, bounds[i])
                ax.plot(x, y, color=c, ls=ls, lw=3.5 if name == "Compositional" else 2.2)
                top = max(top, y.max())
            ax.set_ylim(0, 1.1 * top)
    if args.prior_bounds:
        line = dict(color="0.4", ls="--", lw=1.0, zorder=0)
        for i, b in enumerate(bounds):
            for v in b or ():
                for r in range(i, K):
                    axes[r, i].axvline(v, **line)  # column i: x = parameter i
                for col in range(i):
                    axes[i, col].axhline(v, **line)  # row i: y = parameter i
    # large-valued axes (rho_h, Sigma_D): fold the 1e8/1e9 multiplier into the label instead of the corner offset text
    from matplotlib.ticker import FuncFormatter
    for i, (lo_i, hi_i) in enumerate(ranges):
        e = int(np.floor(np.log10(max(abs(lo_i), abs(hi_i)))))
        if abs(e) < 3:
            continue
        fmt = FuncFormatter(lambda v, _, s=10.0**e: f"{v / s:g}")
        lab = (PARAMS[i][1][:-1] + rf"\,/\,10^{{{e}}}$" if args.no_units
               else PARAMS[i][1].replace(r"\ [", rf"\ [10^{{{e}}}\,", 1))
        axes[K - 1, i].xaxis.set_major_formatter(fmt)
        axes[K - 1, i].set_xlabel(lab, fontsize=args.fontsize)
        if i > 0:
            axes[i, 0].yaxis.set_major_formatter(fmt)
            axes[i, 0].set_ylabel(lab, fontsize=args.fontsize)
    for ax in fig.axes:
        ax.tick_params(labelsize=args.tick_scale * args.fontsize)
    for r in range(1, K):  # horizontal y ticks: rotated ones collide with the larger labels
        axes[r, 0].tick_params(axis="y", labelrotation=0)
        axes[r, 0].yaxis.set_label_coords(-0.45, 0.5)
    for i in range(K):
        axes[K - 1, i].xaxis.set_label_coords(0.5, -args.xlabel_offset)
    handles = [Line2D([0], [0], color=c, ls=ls, label=n) for n, _, c, ls in chains]
    if args.prior_bounds:
        handles.append(Line2D([0], [0], color="0.4", ls="--", label="Prior bounds"))
    if args.rc_inset:
        rc_inset(fig, axes, args.rc_inset, chains, args.fontsize, obs_label=args.obs_label,
                 col=args.inset_col, bottom_row=args.inset_row, top_row=args.inset_top,
                 ts=args.inset_tick_scale or args.tick_scale, lfs=args.inset_label_scale * args.fontsize)
    else:
        fig.legend(handles=handles, loc="upper right", fontsize=0.9 * args.fontsize, frameon=False)
    path = os.path.join(args.run_dir, args.out)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
