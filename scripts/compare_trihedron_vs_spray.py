#!/usr/bin/env python
"""Three-way stream forward-model comparison across a halo-flattening sweep.

For each value of ``q_halo`` the same stream is produced three ways, at identical potential,
progenitor and random seed:

  ``rnbody``     full restricted N-body (``stream_agama_rnbody._rnbody_stream``) -- the ground truth
  ``trihedron``  the Palau & Miralda-Escude (2023) Appendix D remap: ONE N-body run at the fiducial
                 q, frozen into the Frenet-Serret frame of the progenitor orbit (``trihedron.py``),
                 then replayed onto the new orbit. One orbit integration per q, milliseconds.
  ``spray``      particle spray (Chen+2024 release recipe), the project's cheap forward model

The point of the rnbody arm is that it is the only one that measures whether the remap is RIGHT.
``trihedron`` and ``rnbody`` share their Plummer draw (``_rnbody_stream`` samples the progenitor
from ``rng`` before anything potential-dependent touches it), so star *i* corresponds one-to-one
between them and the comparison can be made per star, in kpc. Spray has no such correspondence and
is compared distributionally only.

Fiducial potential: whatever ``--simulator`` names, at the centre of its priors -- by default
``stream_agama_rnbody_cautun_fixed``, i.e. Cautun+(2020) as mapped and validated in this repo
(halo refit to 1.28% median v_circ deviation over 2-60 kpc), with the corrected B&H18 progenitors
and t_end = 4 Gyr. Its q_halo is 1.0, which is the sweep's fiducial.

Caveats
-------
* The sweep varies ``q`` at fixed ``rho``/``a``, so ``M(<r)`` and the rotation curve change with it.
  That is deliberate -- it is how the SBI prior actually moves -- but it means the arms are not
  being compared at fixed enclosed mass, and part of any track shift is a mass effect.
* The solar frame is potential-dependent (``_solar_frame`` uses ``v_circ(R0) + V_Sun``), so the
  progenitor's Galactocentric state is recomputed per q from its FIXED observed ICRS coordinates.
  Both the remap and the ground truth see the same one.
* Pal 5 dissolves completely in this potential (``m_bound_final = 0`` in 100/100 realizations at
  1e4 particles). A dissolved cluster still makes a well-defined template, but its ``t_hat`` labels
  are noisier. M68 survives ~43% of the time.
* Spray has no ``m_bound_final`` and ignores ``a_progenitor``; healthy spray star counts are not
  evidence that a progenitor is viable.

Usage
-----
    uv run python scripts/compare_trihedron_vs_spray.py --check-fiducial
    uv run python scripts/compare_trihedron_vs_spray.py --streams M68 --n-particles 100000

Expensive arms are cached under ``<outdir>/cache`` keyed by (stream, arm, q, n_particles, seed),
so re-running only redraws the figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must precede the hydrabflow imports: simulator auto-discovery otherwise probes for a free GPU,
# which stalls for many minutes on a busy box (and this script is pure CPU/agama anyway).
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trihedron import build_template, remap, auto_window  # noqa: E402
from ppc_summary_statistics import CH, NAMES, binned_median, binned_std  # noqa: E402
from plot_fixed_potential_samples import (  # noqa: E402
    PLX_CH,
    QTY,
    project_sample,
    robust_lim,
    stream_frame,
)

from hydrabflow.simulators.stream_agama import (  # noqa: E402
    _agama,
    _host_potential,
    _solar_frame,
    _spray_stream,
)
from hydrabflow.simulators.stream_agama_rnbody import _rnbody_stream  # noqa: E402
from hydrabflow.simulators.stream_common import sky_projection  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ARM_COLORS = {"rnbody": "tab:red", "trihedron": "tab:blue", "spray": "tab:green"}
ARM_ORDER = ["rnbody", "trihedron", "spray"]
Q_KEY = "q_TwoPowerTriaxial_halo"


# --------------------------------------------------------------------------------------------
# configuration -> one parameter row
# --------------------------------------------------------------------------------------------


def prior_center(spec) -> float:
    """Centre of a prior spec: the constant, the mean, or the midpoint."""
    kind = str(spec["type"])
    pp = [float(v) for v in spec["prior_parameters"]]
    if kind == "identity":
        return pp[0]
    if kind == "normal":
        return pp[0]
    if kind == "uniform":
        return 0.5 * (pp[0] + pp[1])
    raise ValueError(f"unsupported prior type {kind!r}")


def load_simulator(name: str):
    """Instantiate the simulator named by a config, purely to read its resolved params."""
    from hydra import compose, initialize_config_dir
    from hydrabflow.config.schema import register_configs
    from hydrabflow.simulators.registry import get_simulator

    register_configs()
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
        cfg = compose(
            config_name="config", overrides=[f"simulator={name}", "composition=global"]
        )
    return get_simulator(cfg.simulator)


def fiducial_row(sim, stream: str) -> dict:
    """Globals + one stream's locals, each at its prior centre."""
    row = {k: prior_center(v) for k, v in sim._priors_global.items()}
    row.update({k: prior_center(v) for k, v in sim._priors_local[stream].items()})
    return row


def progenitor_state(agama, pot, row: dict):
    """Present-day Galactocentric 6D state of the progenitor, and the row's solar frame.

    Verbatim the conversion ``stream_agama._simulate_one`` performs: the observed ICRS coordinates
    are fixed, but the frame they are converted through depends on the potential.
    """
    frame = _solar_frame(agama, pot, row)
    l0, b0, pml0, pmb0 = agama.transformCelestialCoords(
        agama.fromICRStoGalactic,
        row["ra"] * np.pi / 180,
        row["dec"] * np.pi / 180,
        row["mu_ra_cosdec"],
        row["mu_dec"],
    )
    posvel = np.array(
        agama.getGalactocentricFromGalactic(
            l0, b0, row["r"], pml0 * 4.74, pmb0 * 4.74, row["vr"],
            galcen_distance=frame[0], galcen_v_sun=frame[1:4], z_sun=frame[4],
        )
    )
    return posvel, frame


# --------------------------------------------------------------------------------------------
# the three arms
# --------------------------------------------------------------------------------------------


def run_rnbody(agama, pot, posvel, row, opts, n_particles, seed, time_unit_gyr):
    xv, m_bound = _rnbody_stream(
        agama, pot, posvel,
        mass_sat=row["m_progenitor"],
        radius_sat=row["a_progenitor"] / 1e3,
        time_total=row["t_end"] / time_unit_gyr,
        num_particles=n_particles,
        rng=np.random.default_rng(seed),
        n_updates=int(opts["n_updates"]),
        traj_per_update=int(opts["traj_per_update"]),
        accuracy=float(opts["accuracy"]),
        max_num_steps=float(opts["max_num_steps"]),
        update_max_num_steps=float(opts["update_max_num_steps"]),
    )
    return xv, m_bound


def run_spray(agama, pot, posvel, row, n_particles, seed, time_unit_gyr, method="chen"):
    return _spray_stream(
        agama, pot, posvel,
        mass_sat=row["m_progenitor"],
        radius_sat=row["a_progenitor"] / 1e3,
        time_total=row["t_end"] / time_unit_gyr,
        num_particles=n_particles,
        rng=np.random.default_rng(seed),
        method=method,
    )


# --------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------


def in_window(proj, j):
    from ppc_summary_statistics import WINDOW

    lo_ra, hi_ra, lo_dec, hi_dec = WINDOW[j]
    ra, dec = proj[:, CH["ra"]], proj[:, CH["dec"]]
    return (ra >= lo_ra) & (ra <= hi_ra) & (dec >= lo_dec) & (dec <= hi_dec) & np.isfinite(ra)


def edge_centre_ratio(phi1, frac=0.2):
    """Number density in the outer ``frac`` of the phi1 range (both ends) over the central ``frac``.

    Preferred over the phi1 extent for NGC3201/M68, whose in-window extent is saturated by the
    observation window (CLAUDE.md, freed-t_end entry): a stream that overflows its window piles up
    at the edges, and this ratio sees that while the extent cannot.
    """
    phi1 = phi1[np.isfinite(phi1)]
    if len(phi1) < 10:
        return float("nan")
    lo, hi = phi1.min(), phi1.max()
    span = hi - lo
    if span <= 0:
        return float("nan")
    edge = np.sum((phi1 < lo + frac * span) | (phi1 > hi - frac * span)) / (2.0 * frac)
    mid = np.sum(np.abs(phi1 - 0.5 * (lo + hi)) < 0.5 * frac * span) / frac
    return float(edge / mid) if mid > 0 else float("nan")


def arm_metrics(proj, j, R, edges):
    """Distributional summary of one arm, in the stream's great-circle frame."""
    sel = in_window(proj, j)
    out = {"n_in_window": int(sel.sum())}
    if sel.sum() < 10:
        return out
    p = project_sample(R, proj[sel])
    out["phi1_extent_deg"] = float(np.ptp(p["phi1"]))
    out["edge_centre_ratio"] = edge_centre_ratio(p["phi1"])
    out["phi2_track"] = binned_median(p["phi1"], p["phi2"], edges).tolist()
    out["phi2_std"] = binned_std(p["phi1"], p["phi2"], edges).tolist()
    return out


def per_star_error(proj_a, proj_b, xv_a, xv_b, j, R):
    """Per-star agreement between two arms that share particle identity (trihedron vs rnbody).

    The total ``|dx|`` is decomposed because the two components mean very different things. A
    displacement ALONG the local orbit direction slides a star up and down a track the stream
    already occupies, so it barely changes the observed distribution; a PERPENDICULAR displacement
    moves the stream off its track and is what would bias an inference. The same split shows up
    observationally as ``d phi1`` (along) vs ``d phi2`` (across) in the great-circle frame.
    """
    ok = np.isfinite(xv_a).all(1) & np.isfinite(xv_b).all(1)
    out = {"n_compared": int(ok.sum())}
    if ok.sum() == 0:
        return out
    d = xv_a[ok, 0:3] - xv_b[ok, 0:3]
    v = xv_b[ok, 3:6]
    that = v / np.linalg.norm(v, axis=1, keepdims=True)  # local orbit direction at the truth star
    along = np.sum(d * that, axis=1)
    perp = np.linalg.norm(d - along[:, None] * that, axis=1)
    dx = np.linalg.norm(d, axis=1)
    out["dx_kpc_median"] = float(np.median(dx))
    out["dx_kpc_p90"] = float(np.percentile(dx, 90))
    out["dx_along_kpc_median"] = float(np.median(np.abs(along)))
    out["dx_perp_kpc_median"] = float(np.median(perp))
    out["dx_perp_kpc_p90"] = float(np.percentile(perp, 90))
    sel = ok & in_window(proj_b, j)
    if sel.sum() >= 10:
        pa = project_sample(R, proj_a[ok][in_window(proj_b, j)[ok]])
        pb = project_sample(R, proj_b[sel])
        out["dphi1_deg_median"] = float(np.median(np.abs(pa["phi1"] - pb["phi1"])))
        out["dphi2_deg_median"] = float(np.median(np.abs(pa["phi2"] - pb["phi2"])))
        out["dphi2_deg_p90"] = float(np.percentile(np.abs(pa["phi2"] - pb["phi2"]), 90))
    return out


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulator", default="stream_agama_rnbody_cautun_fixed",
                    help="config supplying the fiducial potential, progenitors and rnbody options")
    ap.add_argument("--streams", nargs="+", default=["M68"])
    ap.add_argument("--q", nargs="+", type=float, default=[0.7, 0.85, 1.0, 1.2, 1.4])
    ap.add_argument("--q-fid", type=float, default=None,
                    help="q the template is built at; default = the config's own value")
    ap.add_argument("--n-particles", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--T-myr", type=float, default=200.0,
                    help="half-width of the orbit window the trihedron is built on")
    ap.add_argument("--n-knots", type=int, default=2001)
    ap.add_argument("--auto-T", action="store_true",
                    help="grow T until <0.1%% of stars pin to the window boundary")
    ap.add_argument("--velocity", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--agama-threads", type=int, default=16,
                    help="agama internal threads; the 1e5-particle rnbody arm parallelizes here")
    ap.add_argument("--outdir", default="data_local/trihedron")
    ap.add_argument("--out", default="trihedron", help="figure filename stem inside --outdir")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--check-fiducial", action="store_true",
                    help="only verify that the remap at q=q_fid reproduces the template exactly")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    cache = outdir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    agama = _agama()
    agama.setNumThreads(int(args.agama_threads))
    tu = agama.getUnits()["time"]
    time_unit_gyr = float(getattr(tu, "value", tu)) / 1e3
    myr_to_agama = 1e-3 / time_unit_gyr  # Myr -> agama time units

    sim = load_simulator(args.simulator)
    pot_cfg = sim._pot_cfg
    opts = sim._rnbody_opts()
    streams = {name: sim.target_streams[name] for name in args.streams}

    # --- real members define the great-circle frames -------------------------------------------
    d = np.load(args.real)
    rsim = d["sim_data_projected"]
    rsim = rsim[0] if rsim.ndim == 4 else rsim
    ram = d["attention_mask"]
    ram = ram[:, 0, :] if ram.ndim == 3 else ram
    rvm = d["vlos_mask"]
    rvm = rvm[:, 0, :] if rvm.ndim == 3 else rvm
    jreal = np.asarray(d["j"]).reshape(-1).astype(int)

    results = {}
    for name, j in streams.items():
        row_fid = fiducial_row(sim, name)
        q_fid = float(args.q_fid) if args.q_fid is not None else row_fid[Q_KEY]
        row_fid[Q_KEY] = q_fid
        print(f"\n=== {name} (j={j})  fiducial q={q_fid:.4g}, {args.n_particles} particles ===")

        pot_fid = _host_potential(agama, row_fid, pot_cfg)
        posvel_fid, frame_fid = progenitor_state(agama, pot_fid, row_fid)

        # --- template ---------------------------------------------------------------------------
        key = f"{name}_rnbody_q{q_fid:.4f}_P{args.n_particles}_s{args.seed}"
        xv_fid, m_bound = load_or_run(
            cache / f"{key}.npz", args.no_cache,
            lambda: run_rnbody(agama, pot_fid, posvel_fid, row_fid, opts,
                               args.n_particles, args.seed, time_unit_gyr),
            label=f"{name} rnbody @ q={q_fid:.4g} (template)",
        )
        T = args.T_myr * myr_to_agama
        if args.auto_T:
            tpl, T, n_knots = auto_window(
                agama, pot_fid, posvel_fid, xv_fid, T, args.n_knots,
                t_max=row_fid["t_end"] / time_unit_gyr, velocity=args.velocity,
            )
            print(f"  auto-T: T = {T / myr_to_agama:.1f} Myr, {n_knots} knots")
        else:
            tpl = build_template(agama, pot_fid, posvel_fid, xv_fid, T, args.n_knots,
                                 velocity=args.velocity)
        print("  template diagnostics: "
              + ", ".join(f"{k}={v:.4g}" for k, v in tpl.diagnostics.items()))
        if tpl.diagnostics["boundary_frac"] > 1e-3:
            print(f"  WARNING: {tpl.diagnostics['boundary_frac']:.1%} of stars pin to the window "
                  f"boundary -- raise --T-myr (or use --auto-T)")

        # --- self-test: the remap at the fiducial must be the identity --------------------------
        xv_check = remap(agama, pot_fid, posvel_fid, tpl)
        ok = np.isfinite(xv_fid).all(1)
        err = float(np.nanmax(np.linalg.norm(xv_check[ok, 0:3] - xv_fid[ok, 0:3], axis=1)))
        print(f"  fiducial round-trip: max |dx| = {err:.3e} kpc")
        if args.check_fiducial:
            assert err < 1e-9, f"remap is not exact at the fiducial: {err:.3e} kpc"
            print("  OK")
            continue

        # --- frames + phi1 binning from the real members ----------------------------------------
        rrow = int(np.where(jreal == j)[0][0])
        mem = ram[rrow].astype(bool)
        R, real = stream_frame(rsim[rrow][mem])
        edges = np.linspace(real["phi1"].min(), real["phi1"].max(), 11)

        per_q = {}
        for q in args.q:
            row = dict(row_fid)
            row[Q_KEY] = float(q)
            pot = _host_potential(agama, row, pot_cfg)
            posvel, frame = progenitor_state(agama, pot, row)

            arms = {"trihedron": remap(agama, pot, posvel, tpl)}
            arms["rnbody"], mb = load_or_run(
                cache / f"{name}_rnbody_q{q:.4f}_P{args.n_particles}_s{args.seed}.npz",
                args.no_cache,
                lambda pot=pot, posvel=posvel, row=row: run_rnbody(
                    agama, pot, posvel, row, opts, args.n_particles, args.seed, time_unit_gyr),
                label=f"{name} rnbody @ q={q:.4g}",
            )
            arms["spray"], _ = load_or_run(
                cache / f"{name}_spray_q{q:.4f}_P{args.n_particles}_s{args.seed}.npz",
                args.no_cache,
                lambda pot=pot, posvel=posvel, row=row: (
                    run_spray(agama, pot, posvel, row, args.n_particles, args.seed,
                              time_unit_gyr), float("nan")),
                label=f"{name} spray  @ q={q:.4g}",
            )

            proj = {}
            for arm, xv in arms.items():
                p = sky_projection(xv[None, ...], np.asarray([frame], dtype=float))[0]
                p[:, PLX_CH] = 1.0 / p[:, PLX_CH]  # distance -> parallax, as the real npz stores
                proj[arm] = p

            m = {arm: arm_metrics(proj[arm], j, R, edges) for arm in ARM_ORDER}
            # Primary accuracy measure: how far each approximation's binned phi2 track sits from
            # the ground truth's. Distributional on purpose -- see per_star_error's docstring.
            truth = np.array(m["rnbody"].get("phi2_track", [np.nan]))
            for arm in ("trihedron", "spray"):
                got = np.array(m[arm].get("phi2_track", [np.nan]))
                m[arm]["track_offset_deg"] = (
                    float(np.nanmedian(np.abs(got - truth))) if got.shape == truth.shape
                    else float("nan")
                )
            m["trihedron_vs_rnbody"] = per_star_error(
                proj["trihedron"], proj["rnbody"], arms["trihedron"], arms["rnbody"], j, R)
            m["m_bound_final"] = float(mb)
            per_q[f"{q:.4g}"] = m
            e = m["trihedron_vs_rnbody"]
            print(f"  q={q:<5.4g} in-window "
                  + " ".join(f"{a}={m[a].get('n_in_window', 0)}" for a in ARM_ORDER)
                  + f" | per-star |dx| med={e.get('dx_kpc_median', np.nan):.3f}"
                  + f" (along {e.get('dx_along_kpc_median', np.nan):.3f},"
                  + f" perp {e.get('dx_perp_kpc_median', np.nan):.3f}) kpc"
                  + f" | track offset vs truth: trihedron={m['trihedron']['track_offset_deg']:.3f}"
                  + f" spray={m['spray']['track_offset_deg']:.3f} deg"
                  + f" | m_bound={mb:.3g}")
            per_q[f"{q:.4g}"]["_proj"] = proj  # kept in memory for the figures, stripped before json

        results[name] = dict(
            j=j, q_fid=q_fid, template=tpl.diagnostics, per_q=per_q,
            R=R.tolist(), real_phi1_range=[float(edges[0]), float(edges[-1])],
        )
        make_figures(name, j, R, real, rvm[rrow][mem], per_q, args, outdir, edges, q_fid)

    if args.check_fiducial:
        return

    # metrics json (drop the in-memory projections)
    clean = {}
    for name, res in results.items():
        res = dict(res)
        res["per_q"] = {q: {k: v for k, v in m.items() if k != "_proj"}
                        for q, m in res["per_q"].items()}
        clean[name] = res
    path = outdir / f"{args.out}_metrics.json"
    path.write_text(json.dumps(clean, indent=2))
    print(f"\nwrote {path}")
    make_error_figure(results, args, outdir)


def load_or_run(path: Path, no_cache: bool, fn, label: str):
    """Run ``fn`` (returning ``(xv, scalar)``) unless a cached npz is present."""
    if path.exists() and not no_cache:
        d = np.load(path)
        print(f"  [cache] {label}")
        return d["xv"], float(d["extra"])
    t0 = time.time()
    print(f"  [run]   {label} ...", flush=True)
    xv, extra = fn()
    np.savez_compressed(path, xv=xv.astype(np.float32), extra=np.float64(extra))
    print(f"  [done]  {label}  ({time.time() - t0:.1f} s)")
    return xv, float(extra)


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def make_figures(name, j, R, real, real_vmask, per_q, args, outdir: Path, edges, q_fid) -> None:
    qs = list(per_q.keys())

    # --- phi2 vs phi1, one row per q, all three arms overlaid ---------------------------------
    fig, axes = plt.subplots(len(qs), 1, figsize=(9, 2.3 * len(qs)), sharex=True, sharey=True,
                             squeeze=False)
    lims = []
    proj_cache = {}
    for qi, q in enumerate(qs):
        proj_cache[q] = {}
        for arm in ARM_ORDER:
            p = per_q[q]["_proj"][arm]
            sel = in_window(p, j)
            proj_cache[q][arm] = project_sample(R, p[sel]) if sel.sum() else None
            if proj_cache[q][arm] is not None:
                lims.append(proj_cache[q][arm]["phi2"])
    ylim = robust_lim(real["phi2"], *lims) if lims else (-5, 5)
    for qi, q in enumerate(qs):
        ax = axes[qi, 0]
        ax.scatter(real["phi1"], real["phi2"], s=6, c="0.6", label="Gaia members", zorder=1)
        for arm in ARM_ORDER:
            p = proj_cache[q][arm]
            if p is None:
                continue
            ax.scatter(p["phi1"], p["phi2"], s=1.5, alpha=0.25, c=ARM_COLORS[arm],
                       label=arm, rasterized=True, zorder=2)
            ax.plot(0.5 * (edges[:-1] + edges[1:]),
                    binned_median(p["phi1"], p["phi2"], edges),
                    c=ARM_COLORS[arm], lw=1.6, zorder=3)
        ax.set_ylim(*ylim)
        ax.set_ylabel("phi2 [deg]")
        ax.text(0.01, 0.93, f"q_halo = {q}", transform=ax.transAxes, va="top", fontsize=9)
        if qi == 0:
            ax.legend(loc="upper right", fontsize=7, markerscale=4, framealpha=0.9)
    axes[-1, 0].set_xlabel("phi1 [deg]")
    fig.suptitle(f"{name}: restricted N-body vs trihedron remap vs particle spray "
                 f"(template built at q={q_fid:.4g}, N={args.n_particles})", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.07, hspace=0.06)
    path = outdir / f"{args.out}_{name}_phi2.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")

    # --- all five observables, one column per q ------------------------------------------------
    fig, axes = plt.subplots(len(QTY), len(qs), figsize=(3.1 * len(qs), 2.0 * len(QTY)),
                             sharex="col", squeeze=False)
    for qi, q in enumerate(qs):
        for ri, (key, label) in enumerate(QTY):
            ax = axes[ri, qi]
            rmask = real_vmask.astype(bool) if key == "vlos" else np.ones(len(real["phi1"]), bool)
            ax.scatter(real["phi1"][rmask], real[key][rmask], s=5, c="k", zorder=5)
            vals = [real[key][rmask]]
            for arm in ARM_ORDER:
                p = proj_cache[q][arm]
                if p is None:
                    continue
                ax.scatter(p["phi1"], p[key], s=1.0, alpha=0.2, c=ARM_COLORS[arm],
                           rasterized=True)
                vals.append(p[key])
            ax.set_ylim(*robust_lim(*vals))
            if qi == 0:
                ax.set_ylabel(label, fontsize=8)
            if ri == 0:
                ax.set_title(f"q = {q}", fontsize=9)
            ax.tick_params(labelsize=7)
        axes[-1, qi].set_xlabel("phi1 [deg]", fontsize=8)
    fig.suptitle(f"{name}: " + "  ".join(f"{a} ({ARM_COLORS[a]})" for a in ARM_ORDER)
                 + "   black = Gaia members", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.05, hspace=0.08, wspace=0.18)
    path = outdir / f"{args.out}_{name}_overlay.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def make_error_figure(results, args, outdir: Path) -> None:
    """How far the remap drifts from the ground truth as the potential moves off the fiducial."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, res in results.items():
        # Plot against q itself, not |q - q_fid|: the remap's accuracy is NOT symmetric about the
        # fiducial (it degrades on the oblate side and stays excellent on the prolate one), which
        # folding the axis would hide.
        qs = np.array([float(q) for q in res["per_q"]])
        order = np.argsort(qs)
        dq = qs
        keys = list(res["per_q"])
        def col(field, arm="trihedron_vs_rnbody"):
            return np.array([res["per_q"][keys[i]][arm].get(field, np.nan) for i in order])

        for arm in ("trihedron", "spray"):
            axes[0].plot(dq[order], col("track_offset_deg", arm),
                         "o-" if arm == "trihedron" else "s--",
                         c=ARM_COLORS[arm], label=f"{name} {arm}")
        axes[1].plot(dq[order], col("dx_kpc_median"), "o-", label=f"{name} |dx| total")
        axes[1].plot(dq[order], col("dx_along_kpc_median"), "^:", alpha=0.7,
                     label=f"{name} along-track")
        axes[1].plot(dq[order], col("dx_perp_kpc_median"), "s--", label=f"{name} perpendicular")
    axes[0].set_xlabel("q_halo")
    axes[0].set_ylabel("median |d phi2 track| vs rnbody [deg]")
    axes[0].set_title("accuracy vs ground truth (distributional)")
    axes[1].set_xlabel("q_halo")
    axes[1].set_ylabel("per-star |dx| vs rnbody [kpc]")
    axes[1].set_title("per-star displacement\n(along-track = star decorrelation, not remap error)")
    for ax in axes:
        for name, res in results.items():
            ax.axvline(res["q_fid"], color="0.5", ls=":", lw=1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / f"{args.out}_error_vs_dq.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
