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


def stack(npz, member):
    cols = []
    for key, _, is_log in PARAMS:
        x = np.asarray(npz[key])[member].reshape(-1)
        cols.append(10.0**x if is_log else x)
    return np.column_stack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="real_global_vs_streams_corner_latex.png")
    ap.add_argument("--curve-only", default=None, help="eval dir run with eval.observed_groups=[vcirc_kms]")
    ap.add_argument("--stream-only", default=None, help="eval dir run with eval.observed_groups=[sim_summary]")
    ap.add_argument("--no-joint", action="store_true", help="drop the per-stream (stream + curve) chains")
    ap.add_argument("--no-global", action="store_true", help="drop the pooled Global chain")
    args = ap.parse_args()

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

    all_data = np.vstack([c[1] for c in chains])
    lo, hi = np.nanpercentile(all_data, [0.5, 99.5], axis=0)
    pad = 0.05 * np.where(hi > lo, hi - lo, 1.0)
    ranges = list(zip(lo - pad, hi + pad))

    fig = None
    for _, data, c, ls in chains:
        fig = corner.corner(
            data, fig=fig, labels=[p[1] for p in PARAMS], range=ranges,
            color=c, bins=40, label_kwargs={"fontsize": 16},
            plot_datapoints=False, plot_density=False, fill_contours=False,
            smooth=1.0, levels=(0.68, 0.95), hist_kwargs={"density": True, "linestyle": ls},
            contour_kwargs={"linestyles": ls},
        )
    handles = [Line2D([0], [0], color=c, ls=ls, label=n) for n, _, c, ls in chains]
    fig.legend(handles=handles, loc="upper right", fontsize=14, frameon=False)
    path = os.path.join(args.run_dir, args.out)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
