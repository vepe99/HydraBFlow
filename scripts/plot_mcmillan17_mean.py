#!/usr/bin/env python
"""Plot the single all-at-the-mean restricted-N-body realization of the three streams in the
McMillan (2017) potential, together with each progenitor's orbit.

Companion to ``conf/simulator/stream_agama_rnbody_mcmillan17_mean.yaml`` (every global potential
parameter AND every local progenitor parameter pinned to its prior mean, 10^4 particles, 5 Gyr),
generated with ``hydrabflow.pipeline.simulate_multistream data.n_simulations=1`` so the grouped npz
holds exactly one realization per stream.

Figures written into ``--out``:

  ``streams_sky.png``          RA/Dec per stream: all simulated particles, the subset inside the
                               observation window, the real Gaia members and the progenitor.
  ``streams_stream_frame.png`` phi2 / parallax / mu_phi1 / mu_phi2 / v_los vs phi1 in each stream's
                               coordinate frame, with the noise-convolved observation-model
                               realization over the raw stream.
  ``streams_icrs_with_orbit.png``  the same, but in the CATALOGUE (ICRS) observables — alpha, delta,
                               parallax, mu_alpha*, mu_delta, v_los — still against phi1, so only the
                               abscissa depends on the frame.
  ``streams_icrs_vs_alpha.png``  pure observation space: delta, parallax, mu_alpha*, mu_delta and
                               v_los against alpha (RA), so NEITHER axis depends on a frame fit.

``--frame streamfinder`` (the DEFAULT) uses the PUBLISHED Ibata+2024 Table 3 pole and RA zero-point
(`scripts/streamfinder_frame.py`), so phi1/phi2 here mean the same thing as in the literature.
``--frame fit`` instead fits a great circle to whichever real member set is loaded
(`ppc_summary_statistics.fit_frame`, which is what the summary-statistics augmentation uses
internally) — its pole, handedness AND zero-point all move with the catalogue, so its absolute phi1
values are not comparable with published figures.
  ``progenitor_orbits.png``    the progenitor orbit that drives the moving Plummer potential:
                               Galactocentric x-y, R-z and r(t) over the full [-t_end, 0] window.
  ``streams_with_orbit.png``   the same five observables vs phi1, with the LAST ``--orbit-gyr``
                               (default 0.3) of that orbit projected into observation space and
                               overplotted as a continuous line wherever it crosses the stream's
                               RA/Dec window, coloured by time.
  ``summary.json``             bound remnant mass, in-window counts, phi1 extents, orbit peri/apo.

The orbit is recomputed here exactly as the simulator does it (rewind the progenitor's present-day
state by t_end with agama, then integrate forward densely — ``stream_agama_rnbody._rnbody_stream``),
using the same host potential and the same solar frame, so it is the trajectory the particles were
actually released along and not an approximation of it.

Usage:
  .venv/bin/python scripts/plot_mcmillan17_mean.py \
    --sim data_local/mcmillan17_mean/mcmillan17_mean.npz \
    --simulator stream_agama_rnbody_mcmillan17_mean \
    --out data_local/mcmillan17_mean
"""

from __future__ import annotations

import argparse
import json
import os

# CPU-only: importing hydrabflow pulls in the JAX backend, which probes the GPUs otherwise.
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ppc_summary_statistics import (  # noqa: E402  (same-dir import)
    CH,
    NAMES,
    WINDOW,
    augment_sim,
    fit_frame,
    project,
)
from streamfinder_frame import frames as streamfinder_frames  # noqa: E402
from streamfinder_frame import mixed_frames  # noqa: E402

DEFAULT_REAL = "assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz"
# sim_data_projected channel 2 is the heliocentric DISTANCE [kpc]; the real npz (and the training
# observation model, via convert_distance_to_parallax) stores a PARALLAX [mas] = 1/d instead.
DIST_CH = 2
QTY = [
    ("phi2", "phi2 [deg]"),
    ("plx", "parallax [mas]"),
    ("mu_phi1", "mu_phi1 [mas/yr]"),
    ("mu_phi2", "mu_phi2 [mas/yr]"),
    ("vlos", "v_los [km/s]"),
]
COLOR = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
# Progenitor sky positions (identical to the `ra`/`dec` identity priors in conf/simulator/stream_agama.yaml);
# only --frame palau needs them, to anchor its phi1 zero-point.
PROGENITOR_RADEC = {"Pal5": (229.022, -0.112), "NGC3201": (154.403, -46.412),
                    "M68": (189.867, -26.744)}
# The same observables in the RAW ICRS coordinates the catalogue is written in, all still plotted
# against phi1 (the along-stream coordinate) so the panels are ordered along the stream.
QTY_ICRS = [
    ("ra", "alpha (RA) [deg]"),
    ("dec", "delta (Dec) [deg]"),
    ("plx", "parallax [mas]"),
    ("pmra", "mu_alpha* [mas/yr]"),
    ("pmdec", "mu_delta [mas/yr]"),
    ("vlos", "v_los [km/s]"),
]
# The same catalogue observables plotted against alpha (RA) itself: pure observation space, with no
# dependence on any great-circle frame on either axis.
QTY_VS_ALPHA = [q for q in QTY_ICRS if q[0] != "ra"]


# --------------------------------------------------------------------------- projection helpers
def frame_of(real_stars):
    return fit_frame(real_stars[:, CH["ra"]], real_stars[:, CH["dec"]])


def to_frame(R, stars, *, parallax_channel: bool):
    """Project (N, 6) observation-space stars into the great-circle frame ``R``."""
    phi1, phi2, mu1, mu2 = project(
        R, stars[:, CH["ra"]], stars[:, CH["dec"]], stars[:, CH["mu_ra"]], stars[:, CH["mu_dec"]]
    )
    d = stars[:, DIST_CH]
    plx = d if parallax_channel else 1.0 / d
    return dict(phi1=phi1, phi2=phi2, plx=plx, mu_phi1=mu1, mu_phi2=mu2, vlos=stars[:, CH["vlos"]])


def contiguous_runs(mask, min_len=2):
    """[(start, stop)] index ranges of each contiguous True run of ``mask``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]] + 1
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_len]


def to_icrs(R, stars, *, parallax_channel: bool):
    """ICRS observables of (N, 6) stars, keyed like :data:`QTY_ICRS`, plus the along-stream phi1."""
    phi1, _, _, _ = project(
        R, stars[:, CH["ra"]], stars[:, CH["dec"]], stars[:, CH["mu_ra"]], stars[:, CH["mu_dec"]]
    )
    d = stars[:, DIST_CH]
    return dict(phi1=phi1, ra=stars[:, CH["ra"]], dec=stars[:, CH["dec"]],
                plx=d if parallax_channel else 1.0 / d,
                pmra=stars[:, CH["mu_ra"]], pmdec=stars[:, CH["mu_dec"]],
                vlos=stars[:, CH["vlos"]])


def in_window(stars, j):
    lo_ra, hi_ra, lo_dec, hi_dec = WINDOW[j]
    ra, dec = stars[:, CH["ra"]], stars[:, CH["dec"]]
    return (ra >= lo_ra) & (ra <= hi_ra) & (dec >= lo_dec) & (dec <= hi_dec)


# ------------------------------------------------------------------------------ progenitor orbit
def progenitor_orbits(params_per_stream, cfg_params, t_end, n_knots=4001):
    """Rewind each progenitor by ``t_end`` and integrate forward, as the simulator does."""
    import agama  # noqa: PLC0415  (worker-style late import; agama sets global units)

    from hydrabflow.simulators.stream_agama import _host_potential, _solar_frame
    from hydrabflow.simulators.stream_common import sky_projection

    agama.setUnits(mass=1, length=1, velocity=1)
    tu = agama.getUnits()["time"]
    time_unit_gyr = float(getattr(tu, "value", tu)) / 1e3

    pot = _host_potential(agama, cfg_params["global"], cfg_params["pot_cfg"])
    out = {}
    for j, p in params_per_stream.items():
        frame = _solar_frame(agama, pot, p)
        l0, b0, pml0, pmb0 = agama.transformCelestialCoords(
            agama.fromICRStoGalactic,
            p["ra"] * np.pi / 180, p["dec"] * np.pi / 180, p["mu_ra_cosdec"], p["mu_dec"],
        )
        xv0 = np.array(agama.getGalactocentricFromGalactic(
            l0, b0, p["r"], pml0 * 4.74, pmb0 * 4.74, p["vr"],
            galcen_distance=frame[0], galcen_v_sun=frame[1:4], z_sun=frame[4],
        ))
        T = t_end / time_unit_gyr
        _, back = agama.orbit(potential=pot, ic=xv0, time=-T, trajsize=2, accuracy=1e-10,
                              maxNumSteps=int(1e6))
        t, orb = agama.orbit(potential=pot, ic=back[-1], time=T, timestart=-T, trajsize=n_knots,
                             accuracy=1e-10, maxNumSteps=int(1e6))
        xv = np.asarray(orb)
        # Same projection the particles went through: this config declares no solar prior, so
        # sky_projection takes its astropy path (frames=None) — see stream_common.sky_projection.
        obs = sky_projection(np.concatenate([xv, xv0[None]])[None])[0]
        out[j] = dict(t_gyr=np.asarray(t) * time_unit_gyr, xv=xv, xv_now=xv0,
                      obs=obs[:-1], obs_now=obs[-1])
    return out


# ------------------------------------------------------------------------------------- figures
def fig_sky(real, sim, noisy, prog, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for ax, j in zip(axes, sorted(NAMES)):
        s, r = sim[j], real[j]
        w = in_window(s, j)
        ax.scatter(s[~w, CH["ra"]], s[~w, CH["dec"]], s=0.4, c="0.82", lw=0,
                   label="simulated (outside window)")
        ax.scatter(s[w, CH["ra"]], s[w, CH["dec"]], s=0.6, c=COLOR[j], lw=0, alpha=0.5,
                   label=f"simulated, in window ({w.sum()})")
        if noisy is not None:
            n = noisy[j]
            ax.scatter(n[:, CH["ra"]], n[:, CH["dec"]], s=14, facecolors="none",
                       edgecolors="tab:red", lw=0.7, label=f"observation model ({len(n)})")
        ax.scatter(r[:, CH["ra"]], r[:, CH["dec"]], s=12, c="k", lw=0,
                   label=f"real Gaia members ({len(r)})")
        p = prog[j]["xv_now"]
        lo_ra, hi_ra, lo_dec, hi_dec = WINDOW[j]
        ax.add_patch(plt.Rectangle((lo_ra, lo_dec), hi_ra - lo_ra, hi_dec - lo_dec,
                                   fill=False, ec="k", ls=":", lw=1.0))
        ax.set_title(NAMES[j])
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.legend(fontsize=7, markerscale=3, loc="best")
        del p
    fig.suptitle("McMillan (2017) potential, restricted N-body, 10^4 particles, t_end = 5 Gyr, "
                 "all local parameters at their prior mean", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_stream_frame(real, sim, noisy, frames, out, real_vmask=None, noisy_vmask=None):
    fig, axes = plt.subplots(len(QTY), 3, figsize=(16, 15), sharex="col")
    for col, j in enumerate(sorted(NAMES)):
        R = frames[j]
        pr = to_frame(R, real[j], parallax_channel=True)
        w = in_window(sim[j], j)
        ps = to_frame(R, sim[j][w], parallax_channel=False)
        pn = to_frame(R, noisy[j], parallax_channel=True) if noisy is not None else None
        # v_los is only data where it was measured; elsewhere the pipelines carry an imputed fill.
        rv = real_vmask[j] if real_vmask is not None else np.ones(len(real[j]), bool)
        nv = noisy_vmask[j] if noisy_vmask is not None else None
        for row, (key, lab) in enumerate(QTY):
            ax = axes[row, col]
            meas = key == "vlos"
            ax.scatter(ps["phi1"], ps[key], s=1.0, c=COLOR[j], lw=0, alpha=0.5,
                       label="simulated (in window, noiseless)")
            if pn is not None:
                k = nv if (meas and nv is not None) else slice(None)
                ax.scatter(pn["phi1"][k], pn[key][k], s=12, facecolors="none",
                           edgecolors="tab:red", lw=0.7,
                           label="observation model" + (" (measured)" if meas else ""))
            k = rv if meas else slice(None)
            ax.scatter(pr["phi1"][k], pr[key][k], s=10, c="k", lw=0,
                       label="real Gaia members" + (" (measured)" if meas else ""))
            ax.set_ylabel(lab)
            if row == 0:
                ax.set_title(NAMES[j])
                ax.legend(fontsize=6, markerscale=3, loc="best")
            elif meas:
                ax.legend(fontsize=6, markerscale=3, loc="best")
            if row == len(QTY) - 1:
                ax.set_xlabel("phi1 [deg]")
            lo, hi = np.nanpercentile(np.concatenate([pr[key][k], ps[key]]), [0.5, 99.5])
            pad = 0.15 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("Stream-frame observables (frame fitted to the real members of each stream)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_orbits(prog, out):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for col, j in enumerate(sorted(NAMES)):
        o = prog[j]
        x, y, z = o["xv"][:, 0], o["xv"][:, 1], o["xv"][:, 2]
        Rcyl = np.hypot(x, y)
        rad = np.sqrt(x**2 + y**2 + z**2)
        c = COLOR[j]
        ax = axes[0, col]
        ax.plot(x, y, lw=0.6, c=c)
        ax.scatter(*o["xv_now"][:2], c="k", s=35, zorder=5, marker="*")
        ax.scatter(o["xv"][0, 0], o["xv"][0, 1], c="tab:red", s=25, zorder=5, marker="o")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_aspect("equal")
        ax.set_title(f"{NAMES[j]}  (star = today, red = {o['t_gyr'][0]:.0f} Gyr)")
        ax = axes[1, col]
        ax.plot(Rcyl, z, lw=0.6, c=c)
        ax.scatter(np.hypot(*o["xv_now"][:2]), o["xv_now"][2], c="k", s=35, zorder=5, marker="*")
        ax.set_xlabel("R [kpc]")
        ax.set_ylabel("z [kpc]")
        ax.set_aspect("equal")
        ax = axes[2, col]
        ax.plot(o["t_gyr"], rad, lw=0.8, c=c)
        ax.axhline(rad.min(), ls=":", c="0.5")
        ax.axhline(rad.max(), ls=":", c="0.5")
        ax.set_xlabel("t [Gyr]  (0 = today)")
        ax.set_ylabel("r [kpc]")
        ax.set_title(f"peri {rad.min():.2f} / apo {rad.max():.2f} kpc")
    fig.suptitle("Progenitor orbits in the fixed McMillan (2017) potential", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_orbit_overlay(real, sim, noisy, frames, prog, out, real_vmask=None, noisy_vmask=None,
                      orbit_gyr=0.3):
    """The five observables vs phi1 for simulation and observation, with the progenitor ORBIT
    overplotted as a continuous line wherever it crosses that stream's observation window.

    Only the LAST ``orbit_gyr`` of the orbit is drawn. Over the full t_end = 5 Gyr each progenitor
    completes ~15-25 radial periods and re-enters the window many times, so the earlier passages are
    a thicket of arcs that say nothing about where the stream is; the tails observable today were
    released along the most recent passage. The line is coloured by time and broken wherever the
    orbit leaves the RA/Dec window, so each drawn segment is one continuous crossing."""
    fig, axes = plt.subplots(len(QTY), 3, figsize=(16, 15), sharex="col")
    lc_ref = None
    for col, j in enumerate(sorted(NAMES)):
        R = frames[j]
        pr = to_frame(R, real[j], parallax_channel=True)
        ps = to_frame(R, sim[j][in_window(sim[j], j)], parallax_channel=False)
        pn = to_frame(R, noisy[j], parallax_channel=True) if noisy is not None else None
        o = prog[j]
        keep = (o["t_gyr"] >= -orbit_gyr) & in_window(o["obs"], j)
        po = to_frame(R, o["obs"], parallax_channel=False)
        t_o = o["t_gyr"]
        runs = contiguous_runs(keep)            # one continuous line per window crossing
        pnow = to_frame(R, o["obs_now"][None], parallax_channel=False)
        rv = real_vmask[j] if real_vmask is not None else np.ones(len(real[j]), bool)
        nv = noisy_vmask[j] if noisy_vmask is not None else None
        for row, (key, lab) in enumerate(QTY):
            ax = axes[row, col]
            meas = key == "vlos"
            ax.scatter(ps["phi1"], ps[key], s=1.0, c="0.7", lw=0, alpha=0.5)
            if pn is not None:
                k = nv if (meas and nv is not None) else slice(None)
                ax.scatter(pn["phi1"][k], pn[key][k], s=10, facecolors="none",
                           edgecolors="tab:red", lw=0.6, alpha=0.7)
            k = rv if meas else slice(None)
            ax.scatter(pr["phi1"][k], pr[key][k], s=10, c="k", lw=0)
            for a, b in runs:
                seg = np.stack([po["phi1"][a:b], po[key][a:b]], axis=-1)
                lc = LineCollection(np.stack([seg[:-1], seg[1:]], axis=1), cmap="viridis",
                                    norm=plt.Normalize(-orbit_gyr, 0.0), linewidths=1.8, zorder=5)
                lc.set_array(0.5 * (t_o[a:b - 1] + t_o[a + 1:b]))
                ax.add_collection(lc)
                lc_ref = lc
            ax.scatter(pnow["phi1"], pnow[key], s=90, marker="*", c="magenta",
                       edgecolors="k", lw=0.5, zorder=6)
            ax.set_ylabel(lab)
            # Keep the x range on the STREAM: the orbit wanders far outside it in phi1.
            ax.set_xlim(*np.nanpercentile(np.concatenate([pr["phi1"], ps["phi1"]]), [0.2, 99.8]))
            if row == 0:
                ax.set_title(NAMES[j])
                handles = [
                    Line2D([], [], ls="", marker="o", ms=3, c="0.7"),
                    Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="tab:red"),
                    Line2D([], [], ls="", marker="o", ms=4, c="k"),
                    Line2D([], [], lw=1.8, c="tab:green"),
                    Line2D([], [], ls="", marker="*", ms=9, c="magenta", mec="k", mew=0.5),
                ]
                ax.legend(handles, [
                    "simulated (in window, noiseless)", "observation model", "real Gaia members",
                    f"progenitor orbit, last {orbit_gyr:g} Gyr (in window)", "progenitor today",
                ], fontsize=6, markerscale=1, loc="best")
            if row == len(QTY) - 1:
                ax.set_xlabel("phi1 [deg]")
            lo, hi = np.nanpercentile(np.concatenate([pr[key][k], ps[key]]), [0.5, 99.5])
            pad = 0.15 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle(f"Observables vs phi1 with the progenitor orbit of the last {orbit_gyr:g} Gyr "
                 "overplotted inside each observation window", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if lc_ref is not None:
        cb = fig.colorbar(lc_ref, ax=axes, orientation="horizontal", fraction=0.02, pad=0.04)
        cb.set_label("orbit time [Gyr]  (0 = today)")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_icrs_overlay(real, sim, noisy, frames, prog, out, real_vmask=None, noisy_vmask=None,
                     orbit_gyr=0.3):
    """The six catalogue (ICRS) observables — alpha, delta, parallax, mu_alpha*, mu_delta, v_los —
    against phi1, for simulation and observation, with the last ``orbit_gyr`` of the progenitor orbit
    overplotted as a continuous line inside the observation window.

    Same content as :func:`fig_orbit_overlay` but in the coordinates the data are measured in rather
    than in the stream frame: only phi1 (the abscissa) is frame-derived, so nothing on the ordinate
    depends on the great-circle fit."""
    fig, axes = plt.subplots(len(QTY_ICRS), 3, figsize=(16, 17), sharex="col")
    lc_ref = None
    for col, j in enumerate(sorted(NAMES)):
        R = frames[j]
        pr = to_icrs(R, real[j], parallax_channel=True)
        ps = to_icrs(R, sim[j][in_window(sim[j], j)], parallax_channel=False)
        pn = to_icrs(R, noisy[j], parallax_channel=True) if noisy is not None else None
        o = prog[j]
        keep = (o["t_gyr"] >= -orbit_gyr) & in_window(o["obs"], j)
        po = to_icrs(R, o["obs"], parallax_channel=False)
        t_o = o["t_gyr"]
        runs = contiguous_runs(keep)
        pnow = to_icrs(R, o["obs_now"][None], parallax_channel=False)
        rv = real_vmask[j] if real_vmask is not None else np.ones(len(real[j]), bool)
        nv = noisy_vmask[j] if noisy_vmask is not None else None
        for row, (key, lab) in enumerate(QTY_ICRS):
            ax = axes[row, col]
            meas = key == "vlos"
            ax.scatter(ps["phi1"], ps[key], s=1.0, c="0.7", lw=0, alpha=0.5)
            if pn is not None:
                k = nv if (meas and nv is not None) else slice(None)
                ax.scatter(pn["phi1"][k], pn[key][k], s=10, facecolors="none",
                           edgecolors="tab:red", lw=0.6, alpha=0.7)
            k = rv if meas else slice(None)
            ax.scatter(pr["phi1"][k], pr[key][k], s=10, c="k", lw=0)
            for a, b in runs:
                seg = np.stack([po["phi1"][a:b], po[key][a:b]], axis=-1)
                lc = LineCollection(np.stack([seg[:-1], seg[1:]], axis=1), cmap="viridis",
                                    norm=plt.Normalize(-orbit_gyr, 0.0), linewidths=1.8, zorder=5)
                lc.set_array(0.5 * (t_o[a:b - 1] + t_o[a + 1:b]))
                ax.add_collection(lc)
                lc_ref = lc
            ax.scatter(pnow["phi1"], pnow[key], s=90, marker="*", c="magenta",
                       edgecolors="k", lw=0.5, zorder=6)
            ax.set_ylabel(lab)
            ax.set_xlim(*np.nanpercentile(np.concatenate([pr["phi1"], ps["phi1"]]), [0.2, 99.8]))
            if row == 0:
                ax.set_title(NAMES[j])
                handles = [
                    Line2D([], [], ls="", marker="o", ms=3, c="0.7"),
                    Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="tab:red"),
                    Line2D([], [], ls="", marker="o", ms=4, c="k"),
                    Line2D([], [], lw=1.8, c="tab:green"),
                    Line2D([], [], ls="", marker="*", ms=9, c="magenta", mec="k", mew=0.5),
                ]
                ax.legend(handles, [
                    "simulated (in window, noiseless)", "observation model", "real Gaia members",
                    f"progenitor orbit, last {orbit_gyr:g} Gyr (in window)", "progenitor today",
                ], fontsize=6, markerscale=1, loc="best")
            if row == len(QTY_ICRS) - 1:
                ax.set_xlabel("phi1 [deg]")
            lo, hi = np.nanpercentile(np.concatenate([pr[key][k], ps[key]]), [0.5, 99.5])
            pad = 0.15 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("ICRS observables vs phi1, with the progenitor orbit of the last "
                 f"{orbit_gyr:g} Gyr overplotted inside each observation window", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if lc_ref is not None:
        cb = fig.colorbar(lc_ref, ax=axes, orientation="horizontal", fraction=0.02, pad=0.04)
        cb.set_label("orbit time [Gyr]  (0 = today)")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def icrs(stars, *, parallax_channel: bool):
    """ICRS observables of (N, 6) stars, keyed like :data:`QTY_VS_ALPHA` (no frame involved)."""
    d = stars[:, DIST_CH]
    return dict(ra=stars[:, CH["ra"]], dec=stars[:, CH["dec"]],
                plx=d if parallax_channel else 1.0 / d,
                pmra=stars[:, CH["mu_ra"]], pmdec=stars[:, CH["mu_dec"]],
                vlos=stars[:, CH["vlos"]])


def fig_icrs_vs_alpha(real, sim, noisy, prog, out, real_vmask=None, noisy_vmask=None,
                      orbit_gyr=0.3):
    """The catalogue observables against alpha (RA): delta, parallax, mu_alpha*, mu_delta, v_los.

    Nothing here depends on a great-circle frame — both axes are measured quantities — so this is
    the rawest view of the three streams, directly comparable with any published sky figure. The
    progenitor orbit of the last ``orbit_gyr`` is overplotted inside each observation window, as in
    :func:`fig_icrs_overlay`."""
    fig, axes = plt.subplots(len(QTY_VS_ALPHA), 3, figsize=(16, 15), sharex="col")
    lc_ref = None
    for col, j in enumerate(sorted(NAMES)):
        pr = icrs(real[j], parallax_channel=True)
        ps = icrs(sim[j][in_window(sim[j], j)], parallax_channel=False)
        pn = icrs(noisy[j], parallax_channel=True) if noisy is not None else None
        o = prog[j]
        keep = (o["t_gyr"] >= -orbit_gyr) & in_window(o["obs"], j)
        po = icrs(o["obs"], parallax_channel=False)
        t_o = o["t_gyr"]
        runs = contiguous_runs(keep)
        pnow = icrs(o["obs_now"][None], parallax_channel=False)
        rv = real_vmask[j] if real_vmask is not None else np.ones(len(real[j]), bool)
        nv = noisy_vmask[j] if noisy_vmask is not None else None
        for row, (key, lab) in enumerate(QTY_VS_ALPHA):
            ax = axes[row, col]
            meas = key == "vlos"
            ax.scatter(ps["ra"], ps[key], s=1.0, c="0.7", lw=0, alpha=0.5)
            if pn is not None:
                k = nv if (meas and nv is not None) else slice(None)
                ax.scatter(pn["ra"][k], pn[key][k], s=10, facecolors="none",
                           edgecolors="tab:red", lw=0.6, alpha=0.7)
            k = rv if meas else slice(None)
            ax.scatter(pr["ra"][k], pr[key][k], s=10, c="k", lw=0)
            for a, b in runs:
                seg = np.stack([po["ra"][a:b], po[key][a:b]], axis=-1)
                lc = LineCollection(np.stack([seg[:-1], seg[1:]], axis=1), cmap="viridis",
                                    norm=plt.Normalize(-orbit_gyr, 0.0), linewidths=1.8, zorder=5)
                lc.set_array(0.5 * (t_o[a:b - 1] + t_o[a + 1:b]))
                ax.add_collection(lc)
                lc_ref = lc
            ax.scatter(pnow["ra"], pnow[key], s=90, marker="*", c="magenta",
                       edgecolors="k", lw=0.5, zorder=6)
            ax.set_ylabel(lab)
            ax.set_xlim(*np.nanpercentile(np.concatenate([pr["ra"], ps["ra"]]), [0.2, 99.8]))
            if row == 0:
                ax.set_title(NAMES[j])
                handles = [
                    Line2D([], [], ls="", marker="o", ms=3, c="0.7"),
                    Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="tab:red"),
                    Line2D([], [], ls="", marker="o", ms=4, c="k"),
                    Line2D([], [], lw=1.8, c="tab:green"),
                    Line2D([], [], ls="", marker="*", ms=9, c="magenta", mec="k", mew=0.5),
                ]
                ax.legend(handles, [
                    "simulated (in window, noiseless)", "observation model", "real Gaia members",
                    f"progenitor orbit, last {orbit_gyr:g} Gyr (in window)", "progenitor today",
                ], fontsize=6, markerscale=1, loc="best")
            if row == len(QTY_VS_ALPHA) - 1:
                ax.set_xlabel("alpha (RA) [deg]")
            lo, hi = np.nanpercentile(np.concatenate([pr[key][k], ps[key]]), [0.5, 99.5])
            pad = 0.15 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("Catalogue observables vs alpha (RA), with the progenitor orbit of the last "
                 f"{orbit_gyr:g} Gyr overplotted inside each observation window", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if lc_ref is not None:
        cb = fig.colorbar(lc_ref, ax=axes, orientation="horizontal", fraction=0.02, pad=0.04)
        cb.set_label("orbit time [Gyr]  (0 = today)")
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="grouped npz from simulate_multistream")
    ap.add_argument("--real", default=DEFAULT_REAL)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--frame", choices=("streamfinder", "palau", "fit"), default="streamfinder",
                    help="phi1/phi2 frame. 'streamfinder' (default) = the PUBLISHED Ibata+2024 "
                         "Table 3 pole and RA zero-point. 'palau' = Palau & Miralda-Escude's "
                         "Appendix A2 construction for M68 (L1 pole + phi1=0 at the stream extreme "
                         "nearest the progenitor -> progenitor at phi1 = -24.1), with Pal5 and "
                         "NGC3201 keeping their published STREAMFINDER frames. 'fit' = a "
                         "great circle fitted to whichever member set is loaded (what the "
                         "summary-statistics augmentation uses); its absolute phi1 is not "
                         "comparable with published figures.")
    ap.add_argument("--palau-loss", choices=("L1", "L2"), default="L1",
                    help="--frame palau: L1 (their R1, minimises the SCATTER of phi2, leaving a "
                         "non-zero mean) or L2 (their R2, mean phi2 ~ 0)")
    ap.add_argument("--simulator", default="stream_agama_rnbody_mcmillan17_mean",
                    help="simulator config name (potential + solar frame for the orbits)")
    ap.add_argument("--aug-preset", default="stream_global_v5",
                    help="training observation model used for the overlay realization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orbit-gyr", type=float, default=0.3,
                    help="only the last N Gyr of the progenitor orbit is overplotted")
    ap.add_argument("--no-noise", action="store_true",
                    help="skip the observation-model realization overlay")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.sim)
    sd = d["sim_data_projected"]              # (1, 3, P, 6)
    jarr = np.asarray(d["j"]).reshape(sd.shape[0], sd.shape[1]).astype(int)
    sim = {int(jarr[0, s]): sd[0, s] for s in range(sd.shape[1])}

    rd = np.load(args.real)
    rsim = rd["sim_data_projected"]
    rsim = rsim[0] if rsim.ndim == 4 else rsim
    ram = rd["attention_mask"]
    ram = ram[:, 0, :] if ram.ndim == 3 else ram
    rj = np.asarray(rd["j"]).reshape(-1).astype(int)
    real = {int(rj[i]): rsim[i][ram[i].astype(bool)] for i in range(rsim.shape[0])}
    rvm = rd["vlos_mask"]
    rvm = rvm[:, 0, :] if rvm.ndim == 3 else rvm
    real_vmask = {int(rj[i]): rvm[i].astype(bool)[ram[i].astype(bool)] for i in range(rsim.shape[0])}
    if args.frame == "streamfinder":
        sf = streamfinder_frames()
        frames = {j: sf[NAMES[j]] for j in real}
    elif args.frame == "palau":
        mf = mixed_frames(loss=args.palau_loss)
        frames = {j: mf[NAMES[j]] for j in real}
    else:
        frames = {j: frame_of(real[j]) for j in real}

    noisy = noisy_vmask = None
    if not args.no_noise:
        aug_sim, aug_mask, aug_vmask = augment_sim(
            sd, jarr, aug_preset=args.aug_preset, seed=args.seed
        )
        noisy = {int(jarr[0, s]): aug_sim[0, s][aug_mask[0, s].astype(bool)]
                 for s in range(sd.shape[1])}
        noisy_vmask = {int(jarr[0, s]): aug_vmask[0, s].astype(bool)[aug_mask[0, s].astype(bool)]
                       for s in range(sd.shape[1])}

    # --- parameters of this realization, straight from the stored draws --------------------
    def col(key):
        a = np.asarray(d[key])
        return a.reshape(a.shape[0], -1)[0] if a.ndim >= 3 else a.reshape(-1)

    local_keys = ["ra", "dec", "r", "vr", "mu_ra_cosdec", "mu_dec",
                  "m_progenitor", "a_progenitor", "t_end"]
    params = {int(jarr[0, s]): {k: float(col(k)[s]) for k in local_keys if k in d.files}
              for s in range(sd.shape[1])}
    glob = {k: float(np.asarray(d[k]).reshape(-1)[0]) for k in d.files
            if k.endswith(("_halo", "_Disk")) and np.asarray(d[k]).size <= sd.shape[0] * 4}

    from hydrabflow.config import register_configs  # noqa: PLC0415
    register_configs()
    from hydra import compose, initialize_config_dir  # noqa: PLC0415

    with initialize_config_dir(config_dir=os.path.abspath("conf"), version_base=None):
        cfg = compose("config", overrides=[f"simulator={args.simulator}"])
    cp = cfg.simulator.params
    pot_cfg = {k: cp[k] for k in ("halo_r_t_kpc", "gas_disks", "thick_disk", "disk_vertical",
                                  "bulge_density_norm", "halo_parameterization") if k in cp}
    prog = progenitor_orbits(params, {"global": glob, "pot_cfg": pot_cfg},
                             t_end=params[0]["t_end"])

    fig_sky(real, sim, noisy, prog, os.path.join(args.out, "streams_sky.png"))
    fig_stream_frame(real, sim, noisy, frames, os.path.join(args.out, "streams_stream_frame.png"),
                     real_vmask=real_vmask, noisy_vmask=noisy_vmask)
    fig_orbits(prog, os.path.join(args.out, "progenitor_orbits.png"))
    fig_orbit_overlay(real, sim, noisy, frames, prog,
                      os.path.join(args.out, "streams_with_orbit.png"),
                      real_vmask=real_vmask, noisy_vmask=noisy_vmask, orbit_gyr=args.orbit_gyr)
    fig_icrs_overlay(real, sim, noisy, frames, prog,
                     os.path.join(args.out, "streams_icrs_with_orbit.png"),
                     real_vmask=real_vmask, noisy_vmask=noisy_vmask, orbit_gyr=args.orbit_gyr)
    fig_icrs_vs_alpha(real, sim, noisy, prog,
                      os.path.join(args.out, "streams_icrs_vs_alpha.png"),
                      real_vmask=real_vmask, noisy_vmask=noisy_vmask, orbit_gyr=args.orbit_gyr)

    # --- numeric summary ------------------------------------------------------------------
    mb = d["m_bound_final"] if "m_bound_final" in d.files else None
    summary = {"potential": "McMillan (2017), fixed", "globals": glob, "streams": {}}
    for j in sorted(sim):
        w = in_window(sim[j], j)
        ps = to_frame(frames[j], sim[j][w], parallax_channel=False)
        pr = to_frame(frames[j], real[j], parallax_channel=True)
        o = prog[j]
        rad = np.linalg.norm(o["xv"][:, :3], axis=1)
        summary["streams"][NAMES[j]] = {
            **params[j],
            "m_bound_final": (float(np.asarray(mb).reshape(sd.shape[0], sd.shape[1], -1)[0, j, 0])
                              if mb is not None else None),
            "n_particles": int(sim[j].shape[0]),
            "n_in_window": int(w.sum()),
            "n_real_members": int(len(real[j])),
            "phi1_extent_sim_deg": float(np.ptp(ps["phi1"])) if w.sum() else None,
            "phi1_extent_real_deg": float(np.ptp(pr["phi1"])),
            "orbit_peri_kpc": float(rad.min()),
            "orbit_apo_kpc": float(rad.max()),
        }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary["streams"], indent=2))


if __name__ == "__main__":
    main()
