#!/usr/bin/env python
"""Compare N realizations of the three streams simulated in ONE fixed potential, in observation
space, against the real Gaia members.

Intended for the diagnostic ``stream_agama_rnbody_cautun_fixed`` config, where every global
potential parameter and every progenitor parameter is an ``identity`` constant, so the only
variation between rows is the four present-day phase-space coordinates of each progenitor
(``vr``, ``r``, ``mu_ra_cosdec``, ``mu_dec``) within their measurement errors. Any spread you see in
these panels is therefore the stream's sensitivity to the progenitor's measured 6D position alone,
with the Galaxy held fixed — not prior spread over the potential.

Figures:

  ``<out>_<stream>_phi2.png``  one PER STREAM: a grid (``--grid-cols``, default 10, so 10x10 for
                        100 samples) with one axis per realization, showing only the across-track
                        coordinate phi2 vs phi1, real Gaia members in grey behind each sample.
  ``<out>_overlay.png`` one axis per (observable, stream): the full available phase space —
                        phi2 / parallax / mu_phi1 / mu_phi2 / v_los vs phi1 — with all N samples
                        overlaid on the real members.

Coordinates are each stream's data-driven great-circle frame — the same frame the summary-statistics
augmentation fits from the REAL members (``ppc_summary_statistics.fit_frame``) — so phi1 runs along
the observed stream and phi2 across it.

By default the samples are pushed through the TRAINING observation model (RA/Dec window ->
member-count subsample -> Gaia magnitudes -> DR3 errors -> noise -> v_los mask) via
``ppc_summary_statistics.augment_sim``, so a panel shows what the network would actually be fed.
``--no-noise`` plots the raw noiseless streams instead (all 1000 particles inside the window).

Usage:
  uv run python scripts/plot_fixed_potential_samples.py \
    --sim data_local/cautun_fixed/cautun_fixed_10.npz \
    --out data_local/cautun_fixed/cautun_fixed \
    --simulator stream_agama_rnbody_cautun_fixed
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppc_summary_statistics import (  # noqa: E402  (same-dir import)
    CH,
    NAMES,
    augment_sim,
    fit_frame,
    project,
    window_subsample,
)

# sim_data_projected channel 2 is the heliocentric DISTANCE [kpc] as produced by sky_projection,
# but the observation model's first step (convert_distance_to_parallax) turns it into a PARALLAX
# [mas] = 1/d, which is also what the real Gaia npz stores. The raw (--no-noise) path applies the
# same 1/d itself, so both modes are plotted against the real parallax channel.
PLX_CH = 2
QTY = [("phi2", "phi2 [deg]"), ("plx", "parallax [mas]"), ("mu_phi1", "mu_phi1 [mas/yr]"),
       ("mu_phi2", "mu_phi2 [mas/yr]"), ("vlos", "vlos [km/s]")]


def stream_frame(rs):
    """Great-circle frame + projected real tracks for one stream's real members."""
    R = fit_frame(rs[:, CH["ra"]], rs[:, CH["dec"]])
    phi1, phi2, m1, m2 = project(
        R, rs[:, CH["ra"]], rs[:, CH["dec"]], rs[:, CH["mu_ra"]], rs[:, CH["mu_dec"]]
    )
    return R, dict(phi1=phi1, phi2=phi2, mu_phi1=m1, mu_phi2=m2, vlos=rs[:, CH["vlos"]],
                   plx=rs[:, PLX_CH])


def project_sample(R, ss):
    phi1, phi2, m1, m2 = project(
        R, ss[:, CH["ra"]], ss[:, CH["dec"]], ss[:, CH["mu_ra"]], ss[:, CH["mu_dec"]]
    )
    return dict(phi1=phi1, phi2=phi2, mu_phi1=m1, mu_phi2=m2, vlos=ss[:, CH["vlos"]],
                plx=ss[:, PLX_CH])


def robust_lim(*arrays, pad=0.15):
    v = np.concatenate([np.asarray(a).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [0.5, 99.5])
    m = pad * max(hi - lo, 1e-9)
    return lo - m, hi + m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="grouped multistream npz (N,S,P,6)")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--out", default="fixed_potential_samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", default="stream_global_ibata_grid",
                    help="augmentation preset supplying the observation model")
    ap.add_argument("--simulator", default="stream_agama_rnbody_cautun_fixed",
                    help="simulator config for the --aug composition (only target_streams matter)")
    ap.add_argument("--no-noise", action="store_true",
                    help="plot the raw noiseless streams instead of the observation-model output")
    ap.add_argument("--grid-cols", type=int, default=0,
                    help="columns in the per-stream phi2 grid; 0 (default) picks a roughly square "
                         "grid from the sample count")
    ap.add_argument("--panel-w", type=float, default=2.4, help="per-panel width [inches]")
    ap.add_argument("--panel-h", type=float, default=2.0, help="per-panel height [inches]")
    ap.add_argument("--dpi", type=int, default=0,
                    help="figure dpi; 0 (default) = 150, stepped to 110 for grids above 120 panels")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- real members define the frames -------------------------------------------------------
    d = np.load(args.real)
    rsim = d["sim_data_projected"]
    rsim = rsim[0] if rsim.ndim == 4 else rsim
    ram = d["attention_mask"]
    ram = ram[:, 0, :] if ram.ndim == 3 else ram
    rvm = d["vlos_mask"]
    rvm = rvm[:, 0, :] if rvm.ndim == 3 else rvm
    jreal = np.asarray(d["j"]).reshape(-1).astype(int)

    # --- samples --------------------------------------------------------------------------------
    ds = np.load(args.sim)
    sd = ds["sim_data_projected"]                      # (N, S, P, 6)
    jsim = np.asarray(ds["j"]).reshape(sd.shape[:2]).astype(int)
    n_samples, n_streams = sd.shape[:2]

    # Which per-stream inputs actually vary between rows — read off the stored draws rather than
    # hardcoded, so a config that frees t_end (or a mass) is described correctly in the titles.
    _local_keys = ("t_end", "m_progenitor", "a_progenitor", "ra", "dec",
                   "vr", "r", "mu_ra_cosdec", "mu_dec")
    # Reduce over the ROW axis only: these arrays are (n_rows, n_streams, 1) and the per-stream
    # constants (m_progenitor, ra, dec, ...) differ BETWEEN streams, so a ptp over the whole array
    # would report every one of them as varying.
    varying = [
        k for k in _local_keys
        if k in ds.files and np.ptp(np.asarray(ds[k]), axis=0).max() > 0
    ]
    varying_label = ", ".join(varying) if varying else "nothing (all inputs fixed)"

    attn = vmask = None
    if args.no_noise:
        # match the observation model's first step so channel 2 is a parallax on both sides
        sd = np.array(sd, dtype=float)
        sd[..., PLX_CH] = 1.0 / sd[..., PLX_CH]
    else:
        sd, attn, vmask = augment_sim(sd, jsim, aug_preset=args.aug, simulator=args.simulator,
                                      seed=args.seed)

    # Precompute per-stream frames, real tracks, sample projections and shared axis limits, so
    # every panel of a stream is on the same scale and the N samples are directly comparable.
    cols = []
    for col in range(n_streams):
        j = int(jsim[0, col])
        rrow = int(np.where(jreal == j)[0][0])
        mem = ram[rrow].astype(bool)
        R, real = stream_frame(rsim[rrow][mem])
        samples, vmeas = [], []
        for i in range(n_samples):
            if attn is None:
                ss = window_subsample(sd[i, col], j, rng)
                keep_v = np.ones(len(ss), bool) if ss is not None else None
            else:
                sel = attn[i, col]
                ss = sd[i, col][sel]
                keep_v = vmask[i, col][sel]
            samples.append(None if ss is None or len(ss) == 0 else project_sample(R, ss))
            vmeas.append(keep_v)
        ok = [s for s in samples if s is not None]
        lims = dict(
            phi1=robust_lim(real["phi1"], *[s["phi1"] for s in ok]),
            **{k: robust_lim(real[k], *[s[k] for s in ok]) for k, _ in QTY},
        )
        cols.append((j, real, samples, vmeas, lims))

    kind = "raw (noiseless)" if args.no_noise else "through the training observation model"

    # ---------------------------------------------------------------------------------------- #
    # Figures 1..n_streams: ONE grid per stream, one axis per sample, phi2 only.
    # ---------------------------------------------------------------------------------------- #
    # Grid shape: default to a roughly square grid so a few hundred realizations stay on one
    # viewable canvas (500 samples -> 23x22 rather than 10x50), overridable with --grid-cols.
    ncol = int(args.grid_cols) if args.grid_cols else max(1, int(np.ceil(np.sqrt(n_samples))))
    nrow = int(np.ceil(n_samples / ncol))
    # Panels are packed edge to edge (shared axes put tick labels only on the outer rows/columns),
    # so nearly the whole canvas is scatter plot. dpi is stepped down for big grids to keep the PNG
    # inside matplotlib's pixel limit and a sane file size.
    dpi = int(args.dpi) if args.dpi else (150 if nrow * ncol <= 120 else 110)
    for j, real, samples, vmeas, lims in cols:
        name = NAMES.get(j, str(j))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(args.panel_w * ncol, args.panel_h * nrow), squeeze=False,
                                 sharex=True, sharey=True,
                                 gridspec_kw=dict(wspace=0.04, hspace=0.04))
        for i in range(nrow * ncol):
            ax = axes[i // ncol][i % ncol]
            if i >= n_samples:
                ax.axis("off")
                continue
            ax.scatter(real["phi1"], real["phi2"], s=1.5, c="0.65", lw=0)
            s = samples[i]
            if s is None:
                ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center",
                        fontsize=6, color="crimson")
            else:
                ax.scatter(s["phi1"], s["phi2"], s=1.5, c="tab:blue", lw=0)
            ax.set_xlim(*lims["phi1"])
            ax.set_ylim(*lims["phi2"])
            ax.tick_params(labelsize=5)
            ax.text(0.02, 0.94, f"#{i}" + (f" n={len(s['phi1'])}" if s is not None else ""),
                    transform=ax.transAxes, fontsize=5, va="top", color="0.25")
        # Label the last axis that actually has content in each column (the final row may be
        # partially filled), and the leftmost axis of every row.
        for k in range(ncol):
            last = nrow - 1 if (nrow - 1) * ncol + k < n_samples else nrow - 2
            if last >= 0:
                axes[last][k].set_xlabel("phi1 [deg]", fontsize=7)
                axes[last][k].tick_params(labelbottom=True)
        for k in range(nrow):
            axes[k][0].set_ylabel("phi2 [deg]", fontsize=7)
        fig.suptitle(f"{name} — {n_samples} realizations in ONE fixed potential (Cautun+2020), "
                     f"{kind}\nvarying per row: {varying_label}; grey = real Gaia members",
                     fontsize=11)
        # NOT tight_layout: it recomputes spacing and would undo the edge-to-edge packing above.
        fig.subplots_adjust(left=0.045, right=0.995, bottom=0.035, top=0.93,
                            wspace=0.04, hspace=0.04)
        out = f"{args.out}_{name}_phi2.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"wrote {out}")

    # ---------------------------------------------------------------------------------------- #
    # Figure 2: all samples overlaid, one axis per (observable, stream)
    # ---------------------------------------------------------------------------------------- #
    fig2, axes2 = plt.subplots(len(QTY), n_streams, figsize=(5.0 * n_streams, 3.0 * len(QTY)),
                               squeeze=False)
    cmap = plt.get_cmap("viridis")
    # With ~100 realizations x a few hundred stars a panel holds tens of thousands of points, so
    # shrink and fade the markers rather than letting the sim saturate the real members.
    pt_size = 3.0 if n_samples <= 12 else 1.2
    pt_alpha = 0.45 if n_samples <= 12 else 0.12
    for col, (j, real, samples, vmeas, lims) in enumerate(cols):
        for r, (key, ylab) in enumerate(QTY):
            ax = axes2[r][col]
            rmask = slice(None)
            if key == "vlos":  # real v_los exists only for the measured subset
                rrow = int(np.where(jreal == j)[0][0])
                mem = ram[rrow].astype(bool)
                rmask = (rvm[rrow].astype(bool) & mem)[mem]
            ax.scatter(real["phi1"][rmask], real[key][rmask], s=10, c="k", lw=0, zorder=3,
                       label="real Gaia")
            for i, s in enumerate(samples):
                if s is None:
                    continue
                sel = slice(None)
                if key == "vlos" and vmeas[i] is not None:
                    sel = vmeas[i]
                ax.scatter(s["phi1"][sel], s[key][sel], s=pt_size, lw=0, alpha=pt_alpha,
                           color=cmap(i / max(n_samples - 1, 1)),
                           label=(f"sample {i}" if r == 0 and n_samples <= 12 else None))
            ax.set_xlim(*lims["phi1"])
            ax.set_ylim(*lims[key])
            ax.set_ylabel(ylab, fontsize=9)
            if r == 0:
                ax.set_title(NAMES.get(j, j), fontsize=12)
            if r == len(QTY) - 1:
                ax.set_xlabel("phi1 [deg]", fontsize=10)
    axes2[0][0].legend(fontsize=6, markerscale=2, ncol=2, loc="best")
    fig2.suptitle(f"All {n_samples} realizations overlaid (varying: {varying_label}) — "
                  f"fixed Cautun+2020 potential, "
                  f"{kind}; black = real Gaia members", fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    fig2.savefig(f"{args.out}_overlay.png", dpi=130)
    print(f"wrote {args.out}_overlay.png")


if __name__ == "__main__":
    main()
