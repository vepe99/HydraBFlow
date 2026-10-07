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
    args = ap.parse_args()
    global _RUN_DIR
    _RUN_DIR = args.run_dir
    if args.halo_agama:
        i = [k for k, _, _ in PARAMS].index("log10_M200_TwoPowerTriaxial_halo")
        PARAMS[i:i + 2] = HALO_PARAMS if args.halo_agama == "log" else [
            ("r_h", r"$r_h\ [{\rm kpc}]$", False), ("rho_h", r"$\rho_h\ [M_\odot\,{\rm kpc^{-3}}]$", False)]
    bounds = prior_bounds(args.run_dir)

    glob = np.load(os.path.join(args.run_dir, "posterior.npz"))
    single = np.load(os.path.join(args.run_dir, "single_stream_posterior.npz"))
    colors = list(plt.cm.RdYlBu_r(np.linspace(0, 1, 1 + len(STREAMS))))
    # (label, samples, colour, linestyle)
    chains = [] if args.no_global else [("Global", stack(glob, 0), colors[0], "-")]
    if not args.no_joint:
        chains += [(s, stack(single, i), colors[1 + i], "-") for i, s in enumerate(STREAMS)]
    if args.stream_only:
        so = np.load(os.path.join(args.stream_only, "single_stream_posterior.npz"))
        chains += [(f"{s} (stream only)", stack(so, i), colors[1 + i], ":") for i, s in enumerate(STREAMS)]
    if args.curve_only:
        co = np.load(os.path.join(args.curve_only, "single_stream_posterior.npz"))
        chains.append(("Rotation curve only", stack(co, 0), "black", "--"))
    if args.modalities_style:
        chains = [(n, d, "#984ea3" if n.startswith("Rotation") else c, "-") for n, d, c, _ in chains]
        if not args.no_global:
            chains = chains[1:] + chains[:1]  # Global drawn on top
    if args.palette == "okabe-ito":  # Okabe & Ito (2008); Global black so the pooled result reads first
        pal = {"Pal5": "#E69F00", "NGC3201": "#0072B2", "M68": "#CC79A7", "Rotation": "#009E73", "Global": "black"}
        chains = [(n, d, next((v for k, v in pal.items() if n.startswith(k)), c), ls) for n, d, c, ls in chains]
    thick = lambda n, a, b: {} if not args.modalities_style else {a: b if n == "Global" else 1.0}  # noqa: E731

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
            color=c, bins=40, label_kwargs={"fontsize": 16},
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
                ax.plot(x, y, color=c, ls=ls, lw=2.0 if name == "Global" else 1.2)
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
    handles = [Line2D([0], [0], color=c, ls=ls, label=n) for n, _, c, ls in chains]
    if args.prior_bounds:
        handles.append(Line2D([0], [0], color="0.4", ls="--", label="Prior bounds"))
    fig.legend(handles=handles, loc="upper right", fontsize=14, frameon=False)
    path = os.path.join(args.run_dir, args.out)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
