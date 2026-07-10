"""Prior-predictive checks for the Ibata (2023) ancillary potential observables (offline helper).

Consumes a flat ``simulate`` dataset (or a single ``<dataset>.chunks/chunk_NNNNN.npz``) generated
with ``simulator=stream_agama_ibata`` and asks whether the prior, pushed through the potential,
brackets the real ancillary measurements on three axes:

  (1) HI terminal velocity v_term(l): percentile band of the stored model ``vterm_kms`` curves vs
      the observed first-quadrant curve (``assets/terminal_velocity.csv``, McClure-Griffiths &
      Dickey 2016). Reports the fraction of observed longitudes inside the prior 5-95% band.
  (2) local surface density Sigma(z=1.1 kpc): histogram of the stored model ``sigma_z`` vs the
      Kuijken & Gilmore (1991) datum 71 +/- 6 Msun/pc^2.
  (3) vertical stellar-density profile rho(z): percentile band of the stored model ``rho_z``,
      shown as SHAPE (each row normalized at its first z bin, since the normalization is free /
      marginalized at inference time). No real overlay yet (Ibata 2017b Fig. 12f is a digitized
      TODO, see new_constrains.md); the band just shows the prior spread.

The stored observables are the noise-free MODEL values (the observational-error resampling and the
log10 of rho are training-time augmentations, not stored), so this PPC works directly on the npz.

Usage:
    python scripts/ppc_ancillary_observables.py DATASET.npz OUT_DIR [--tv assets/terminal_velocity.csv]
        [--max-rows 5000]

Writes ``ppc_ancillary_observables.png`` and ``ppc_ancillary_summary.json`` to OUT_DIR and prints
the JSON summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hydrabflow.simulators.stream_common import (  # noqa: E402
    RHO_Z_KPC,
    SIGMA_Z_ERR_MSUN_PC2,
    SIGMA_Z_OBS_MSUN_PC2,
    VTERM_L_DEG,
)

_DEFAULT_TV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "terminal_velocity.csv"
)


def _load_2d(data: dict, key: str) -> np.ndarray | None:
    """Return a stored observable as (n_rows, n_bins), or None if absent."""
    if key not in data:
        return None
    arr = np.asarray(data[key], dtype=float)
    return arr.reshape(arr.shape[0], -1)


def _band(ax, x, rows, color, label, log=False):
    """Median + 68/95% percentile band of `rows` (n, len(x)) vs x."""
    lo95, lo68, med, hi68, hi95 = np.nanpercentile(rows, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(x, lo95, hi95, color=color, alpha=0.15, lw=0)
    ax.fill_between(x, lo68, hi68, color=color, alpha=0.30, lw=0)
    ax.plot(x, med, color=color, lw=2, label=label)
    if log:
        ax.set_yscale("log")
    return np.vstack([lo95, hi95])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("out_dir")
    ap.add_argument("--tv", default=_DEFAULT_TV, help="observed terminal-velocity CSV")
    ap.add_argument("--max-rows", type=int, default=5000, help="subsample rows for the bands")
    ap.add_argument("--sim-multistream", default=None,
                    help="grouped multistream npz -> also plot the per-stream summary-statistic "
                         "tracks (phi2/mu_phi1/mu_phi2/vlos vs the real Gaia streams)")
    ap.add_argument("--real", default=None, help="observed-streams npz for the summary tracks "
                    "(defaults to the in-repo assets/gaia file)")
    ap.add_argument("--n-sim", type=int, default=40, help="sim streams overlaid in the tracks plot")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw = np.load(args.dataset, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    vterm = _load_2d(data, "vterm_kms")
    sigma_z = _load_2d(data, "sigma_z")
    rho_z = _load_2d(data, "rho_z")
    present = [n for n, a in (("vterm", vterm), ("sigma_z", sigma_z), ("rho_z", rho_z)) if a is not None]
    if not present:
        raise SystemExit(
            "No ancillary observables (vterm_kms/sigma_z/rho_z) in the dataset. Generate it with "
            "simulator=stream_agama_ibata (params.ancillary_observables set)."
        )

    rng = np.random.default_rng(0)

    def sub(arr):
        if arr is None or arr.shape[0] <= args.max_rows:
            return arr
        return arr[rng.choice(arr.shape[0], args.max_rows, replace=False)]

    vterm, sigma_z, rho_z = sub(vterm), sub(sigma_z), sub(rho_z)
    summary: dict = {"dataset": args.dataset, "n_rows": int(raw[list(raw.files)[0]].shape[0])}

    n_panels = len(present)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.6), squeeze=False)
    axes = axes[0]
    ax_i = 0

    # (1) terminal velocity ----------------------------------------------------------------
    if vterm is not None:
        ax = axes[ax_i]; ax_i += 1
        lx = VTERM_L_DEG
        if vterm.shape[1] != len(lx):
            lx = np.arange(vterm.shape[1])  # unknown grid; index axis
        band = _band(ax, lx, vterm, "C0", "prior model")
        try:
            # Filter comment lines first: the header comments contain commas, which confuses
            # genfromtxt's header/column detection. What remains is header + data.
            import io
            body = "".join(ln for ln in open(args.tv) if not ln.lstrip().startswith("#"))
            tv = np.genfromtxt(io.StringIO(body), delimiter=",", names=True)
            lo, hi = band[0], band[1]
            # interpolate band edges to observed longitudes
            oin = (tv["vterm_kms"] >= np.interp(tv["l_deg"], lx, lo)) & (
                tv["vterm_kms"] <= np.interp(tv["l_deg"], lx, hi)
            )
            ax.errorbar(tv["l_deg"], tv["vterm_kms"], yerr=tv["sigma_kms"], fmt="o", ms=4,
                        color="k", capsize=2, label="observed (MGD16)", zorder=5)
            summary["vterm_frac_obs_in_5_95"] = float(np.mean(oin))
        except Exception as exc:  # noqa: BLE001
            summary["vterm_obs_error"] = str(exc)
        ax.set_xlabel("Galactic longitude l [deg]"); ax.set_ylabel(r"$v_{\rm term}$ [km/s]")
        ax.set_title("HI terminal velocity"); ax.legend(fontsize=8)

    # (2) surface density ------------------------------------------------------------------
    if sigma_z is not None:
        ax = axes[ax_i]; ax_i += 1
        vals = sigma_z.reshape(-1)
        ax.hist(vals, bins=60, color="C1", alpha=0.8, density=True)
        ax.axvline(SIGMA_Z_OBS_MSUN_PC2, color="k", lw=2, label="obs 71")
        ax.axvspan(SIGMA_Z_OBS_MSUN_PC2 - SIGMA_Z_ERR_MSUN_PC2,
                   SIGMA_Z_OBS_MSUN_PC2 + SIGMA_Z_ERR_MSUN_PC2, color="k", alpha=0.15)
        ax.set_xlabel(r"$\Sigma(1.1\,{\rm kpc})$ [M$_\odot$/pc$^2$]"); ax.set_ylabel("density")
        ax.set_title("Local surface density"); ax.legend(fontsize=8)
        summary["sigma_z_median"] = float(np.median(vals))
        summary["sigma_z_5_95"] = [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]
        summary["sigma_z_frac_within_obs_1sigma"] = float(
            np.mean(np.abs(vals - SIGMA_Z_OBS_MSUN_PC2) <= SIGMA_Z_ERR_MSUN_PC2)
        )

    # (3) vertical density profile (shape) -------------------------------------------------
    if rho_z is not None:
        ax = axes[ax_i]; ax_i += 1
        zx = RHO_Z_KPC if rho_z.shape[1] == len(RHO_Z_KPC) else np.arange(rho_z.shape[1])
        shape = rho_z / rho_z[:, [0]]  # free normalization -> compare shape, anchored at z[0]
        _band(ax, zx, shape, "C2", "prior model (shape)", log=True)
        ax.set_xlabel("z [kpc]"); ax.set_ylabel(r"$\rho(z)/\rho(z_0)$")
        ax.set_title("Vertical stellar density (shape; real=TODO)"); ax.legend(fontsize=8)
        summary["rho_z_shape_median_at_zmax"] = float(np.nanmedian(shape[:, -1]))

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "ppc_ancillary_observables.png")
    fig.savefig(out_png, dpi=130)
    summary["figure"] = out_png
    summary["observables_present"] = present

    # Stream-dependent summary statistics: reuse the standalone stream-frame track renderer so the
    # per-stream (phi2, mu_phi1, mu_phi2, vlos) tracks are checked in the same PPC as the ancillary
    # potential observables (user request). Needs a grouped multistream npz (the (N,S,P,6) test set).
    if args.sim_multistream:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ppc_summary_statistics import render as _render_tracks

        real = args.real
        if real is None:
            cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "assets", "gaia", "gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
            real = cand if os.path.exists(cand) else None
        if real and os.path.exists(real):
            tracks_png = os.path.join(args.out_dir, "ppc_stream_summary_statistics.png")
            try:
                _render_tracks(real, args.sim_multistream, tracks_png, n_sim=args.n_sim)
                summary["stream_summary_tracks_figure"] = tracks_png
            except Exception as exc:  # noqa: BLE001
                summary["stream_summary_tracks_error"] = str(exc)
        else:
            summary["stream_summary_tracks_error"] = "no real observed-streams npz found (--real)"

    with open(os.path.join(args.out_dir, "ppc_ancillary_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
