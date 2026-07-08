"""Overlay two particle-spray recipes (Fardal+2015 vs Chen+2024) against the real Gaia members.

Offline validation helper for the ``spray_method`` seam in ``stream_agama``. Give it two flat
simulate datasets generated from the SAME seed (so the potentials and progenitor phase-space
draws are identical and only the release recipe differs) and it plots, per target stream, the
sky (RA/Dec) and proper-motion distributions of each recipe with the observed Gaia members
overlaid. Also reports each recipe's "real-locus reach" (fraction of real members with >=1
simulated particle within a sky + PM tolerance), reusing the PPC definition.

Usage:
    python scripts/compare_spray_methods.py FARDAL.npz CHEN.npz OUT.png [--real PATH]
        [--sky-tol-deg 2.0] [--pm-tol-masyr 1.5]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_prior_predictive import (  # noqa: E402
    DEFAULT_REAL,
    STREAM_NAMES,
    _reach,
    load_dataset,
    real_members,
)


def _stream_xy_pm(sp: np.ndarray, jarr: np.ndarray, jj: int):
    """(sky RA*cos(dec), Dec) and (pm_ra_cosdec, pm_dec) for the finite particles of stream ``jj``."""
    feats = sp[jarr == jj].reshape(-1, 6)
    feats = feats[np.isfinite(feats).all(axis=1)]  # drop NaN particles (invalid seeds)
    ra, dec, _, pmra, pmdec, _ = feats.T
    xy = np.column_stack([ra * np.cos(np.deg2rad(dec)), dec])
    pm = np.column_stack([pmra, pmdec])
    return ra, dec, pmra, pmdec, xy, pm


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Fardal vs Chen spray against real members.")
    ap.add_argument("fardal", help="flat dataset generated with spray_method=fardal")
    ap.add_argument("chen", help="flat dataset generated with spray_method=chen (same seed)")
    ap.add_argument("out", help="output figure path (.png)")
    ap.add_argument("--real", default=DEFAULT_REAL)
    ap.add_argument("--sky-tol-deg", type=float, default=2.0)
    ap.add_argument("--pm-tol-masyr", type=float, default=1.5)
    args = ap.parse_args()

    fardal = load_dataset(args.fardal)
    chen = load_dataset(args.chen)
    real = real_members(args.real)

    recipes = [("Fardal+2015", fardal, "C0"), ("Chen+2024", chen, "C2")]
    present = [jj for jj in STREAM_NAMES
               if any((d["j"].reshape(-1).astype(int) == jj).any() for _, d, _ in recipes)]

    fig, axes = plt.subplots(len(present), 2, figsize=(12, 4.4 * len(present)), squeeze=False)
    for row, jj in enumerate(present):
        rm = real.get(jj, np.empty((0, 6)))
        ax_sky, ax_pm = axes[row][0], axes[row][1]
        for label, d, color in recipes:
            jarr = d["j"].reshape(-1).astype(int)
            if not (jarr == jj).any():
                continue
            ra, dec, pmra, pmdec, xy, pm = _stream_xy_pm(d["sim_data_projected"], jarr, jj)
            reach = _reach(
                xy, pm,
                np.column_stack([rm[:, 0] * np.cos(np.deg2rad(rm[:, 1])), rm[:, 1]]) if len(rm) else np.empty((0, 2)),
                rm[:, 3:5] if len(rm) else np.empty((0, 2)),
                args.sky_tol_deg, args.pm_tol_masyr,
            )
            ax_sky.scatter(ra, dec, s=1, alpha=0.04, color=color, rasterized=True,
                           label=f"{label} (reach {reach:.0%})")
            ax_pm.scatter(pmra, pmdec, s=1, alpha=0.04, color=color, rasterized=True, label=label)
        if len(rm):
            ax_sky.scatter(rm[:, 0], rm[:, 1], s=6, color="crimson", zorder=5, label="real (Gaia)")
            ax_pm.scatter(rm[:, 3], rm[:, 4], s=6, color="crimson", zorder=5, label="real (Gaia)")
        ax_sky.set_xlabel("RA [deg]"); ax_sky.set_ylabel("Dec [deg]")
        ax_sky.set_title(f"{STREAM_NAMES[jj]} (j={jj}) - sky")
        ax_sky.legend(fontsize=7, markerscale=4, loc="best")
        ax_pm.set_xlabel(r"$\mu_\alpha\cos\delta$ [mas/yr]"); ax_pm.set_ylabel(r"$\mu_\delta$ [mas/yr]")
        ax_pm.set_title(f"{STREAM_NAMES[jj]} - proper motion")
        ax_pm.legend(fontsize=7, markerscale=4, loc="best")
    fig.suptitle(
        f"Particle-spray recipe comparison vs real Gaia members "
        f"(reach tol: {args.sky_tol_deg} deg, {args.pm_tol_masyr} mas/yr)", y=1.002)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
