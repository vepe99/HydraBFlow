#!/usr/bin/env python
"""Three-way stream forward-model comparison across a halo-parameter sweep.

The swept parameter is ``--param`` (default ``q_TwoPowerTriaxial_halo``, the halo flattening;
``rho_TwoPowerTriaxial_halo`` sweeps halo mass, reported as M200 alongside). For each value the
same stream is produced three ways, at identical potential, progenitor and random seed:

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
* The q sweep varies ``q`` at fixed ``rho``/``a``, so ``M(<r)`` and the rotation curve change with
  it; the rho sweep likewise changes M200 AND concentration together (r200 grows with mass while
  ``a`` is held). That is deliberate -- it is how the SBI prior actually moves -- but neither sweep
  isolates one physical quantity, and ``halo_m200`` reports M200/c200 per value so the mass sweep
  can at least be read on a physical axis.
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
    uv run python scripts/compare_trihedron_vs_spray.py --streams Pal5 NGC3201 M68 \
        --param rho_TwoPowerTriaxial_halo --values 1.36e7 1.90e7 2.719573e7 3.81e7 5.44e7 \
        --noise --noise-realizations 120 --out rho

Output goes to ``<outdir>/<stream>/`` (figures + ``<out>_metrics.json``), with the cross-stream
error figure at ``<outdir>/``. Expensive arms are cached under ``<outdir>/<stream>/cache`` keyed by
(arm, swept value, n_particles, seed); the untouched fiducial row is tagged ``fid`` and so is shared
between sweeps of different parameters.
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

from scipy.ndimage import gaussian_filter  # noqa: E402

from trihedron import build_template, remap, auto_window  # noqa: E402
from ppc_summary_statistics import (  # noqa: E402
    CH,
    NAMES,
    WINDOW,
    augment_sim,
    binned_median,
    binned_std,
)
from plot_fixed_potential_samples import (  # noqa: E402
    PLX_CH,
    QTY,
    project_sample,
    robust_lim,
    stream_frame,
)

from hydrabflow.simulators.stream_agama import (  # noqa: E402
    _agama,
    _halo_params,
    _halo_params_m200c,
    _host_potential,
    _resolve_pot_cfg,
    _solar_frame,
    _spray_stream,
)
from hydrabflow.simulators.stream_agama_rnbody import _rnbody_stream  # noqa: E402
from hydrabflow.simulators.stream_common import G_KPC_KMS2_MSUN, sky_projection  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ARM_COLORS = {"rnbody": "tab:red", "trihedron": "tab:blue", "spray": "tab:green"}
ARM_ORDER = ["rnbody", "trihedron", "spray"]
Q_KEY = "q_TwoPowerTriaxial_halo"
H0_KMS_MPC = 70.4  # the cosmology stream_agama._halo_params_m200c assumes, kept in step with it


def param_slug(param: str) -> str:
    """Short filename-safe tag for a swept parameter (``q_TwoPowerTriaxial_halo`` -> ``q``)."""
    return param.split("_")[0]


def arm_cache_name(name: str, arm: str, tag: str, args) -> str:
    """Cache filename for one expensive arm.

    The swept parameter's slug is part of the tag: without it a mass sweep would silently reuse a
    flattening sweep's files. The one exception is the untouched fiducial row, which is the same
    potential whatever parameter is being swept, so it is tagged ``fid`` and shared between sweeps.
    """
    return f"{name}_{arm}_{tag}_P{args.n_particles}_s{args.seed}.npz"


def value_tag(slug: str, value: float, v_fid: float, args) -> str:
    """``fid`` for the config's own fiducial row, else ``<slug><value>`` (``.6g``: rho is ~2.7e7)."""
    if args.fid is None and float(value) == float(v_fid):
        return "fid"
    return f"{slug}{value:.6g}"


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
    from hydrabflow.config import register_configs
    from hydrabflow.registry import get_simulator

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


def halo_m200(agama, row: dict, pot_cfg) -> dict:
    """(M200, r200, c200) of this row's halo, so a ``rho`` sweep can be read as a mass sweep.

    Inverse of ``stream_agama._halo_params_m200c``, which only maps (M200, c') -> (rho, a). Same
    cosmology (H0 = 70.4, Delta = 200 x rho_crit) so the two are consistent, and the concentration
    uses that function's own convention ``r_h = r200 / (c200 (2 - gamma))``.
    """
    cfg = _resolve_pot_cfg(pot_cfg)
    if str(cfg["halo_parameterization"]) == "m200_c":
        d = _halo_params_m200c(agama, row, cfg)
    else:
        d = _halo_params(row, float(cfg["halo_r_t_kpc"]))
    halo = agama.Potential(**d)
    h0 = H0_KMS_MPC / 1e3  # km/s/kpc
    rho_crit = 3.0 * h0 * h0 / (8.0 * np.pi * G_KPC_KMS2_MSUN)
    target = (4.0 / 3.0) * np.pi * 200.0 * rho_crit

    def f(r):
        return float(halo.enclosedMass(r)) - target * r**3

    lo, hi = 10.0, 1000.0
    if f(lo) < 0 or f(hi) > 0:
        return {"m200_msun": float("nan"), "r200_kpc": float("nan"), "c200": float("nan")}
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    r200 = 0.5 * (lo + hi)
    gamma = float(d.get("gamma", 1.0))
    r_h = float(d["scaleRadius"])
    return {
        "m200_msun": float(halo.enclosedMass(r200)),
        "r200_kpc": float(r200),
        "c200": float(r200 / (r_h * (2.0 - gamma))),
    }


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


def project_xv(xv, frame):
    """``sky_projection`` for one arm, with non-finite rows held out.

    agama returns an ALL-NaN projection if a single input row is non-finite, so at 1e6 particles two
    NaN stars (0.0002% -- routine for spray, and for a dissolved progenitor) silently zero the whole
    arm. Projecting the finite subset and writing NaN back keeps the row indexing, which
    ``per_star_error`` relies on to pair trihedron stars with their rnbody counterparts.
    """
    out = np.full((len(xv), 6), np.nan)
    ok = np.isfinite(xv).all(axis=1)
    if ok.any():
        out[ok] = sky_projection(xv[ok][None, ...], np.asarray([frame], dtype=float))[0]
    return out


def in_window(proj, j):
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


def observed_sample(proj_raw, j, args, seed):
    """The stars an observer would actually have, as ``(sample (n,6), vlos_mask (n,))``.

    ``proj_raw`` is straight out of ``sky_projection``, so channel 2 is heliocentric DISTANCE.

    Without ``--noise`` this is just the RA/Dec window cut, with the ``1/d`` conversion applied so
    channel 2 is a parallax on both the simulated and the real side (the real npz stores parallax).

    With ``--noise`` the stream is pushed through the training observation model
    (window -> subsample to the observed member count -> Gaia magnitudes -> DR3 errors -> noise ->
    v_los mask). That chain cuts each realization down to the observed member count (297 for M68),
    which is far too few to contour, so ``--noise-realizations`` independent realizations of the
    SAME underlying stream are drawn and pooled. Each one individually is exactly what the network
    is fed; pooling them traces the noise-convolved distribution the members are drawn from.

    Tiling every particle into every realization is what the array size is set by, and at 1e6
    particles x 120 realizations that is ~6 GB before the chain even starts. Since the chain keeps
    only ``observed_n_stars`` (297 for M68) per realization at random, drawing each realization from
    an independent uniform subsample of ``--noise-subsample`` particles is statistically the same
    thing at a fraction of the memory. The subsample indices are drawn from ``seed``, which is the
    same for every arm, so the arms stay comparable star for star.
    """
    if not args.noise:
        p = proj_raw[in_window(proj_raw, j)].copy()
        p[:, PLX_CH] = 1.0 / p[:, PLX_CH]
        return p, np.ones(len(p), bool)
    n_real = int(args.noise_realizations)
    n_sub = int(args.noise_subsample)
    if 0 < n_sub < len(proj_raw):
        rng = np.random.default_rng(seed)
        idx = np.stack([rng.choice(len(proj_raw), size=n_sub, replace=False)
                        for _ in range(n_real)])
        sd = proj_raw[idx][:, None, :, :]  # (R, 1, n_sub, 6)
    else:
        sd = np.repeat(proj_raw[None, None, ...], n_real, axis=0)  # (R, 1, P, 6)
    jj = np.full((n_real, 1), j, dtype=int)
    sim, attn, vmask = augment_sim(
        sd, jj, aug_preset=args.aug, simulator=args.simulator, seed=seed
    )
    keep = attn[:, 0].astype(bool)
    return sim[:, 0][keep], vmask[:, 0][keep].astype(bool)


def arm_metrics(sample, R, edges):
    """Distributional summary of one arm, in the stream's great-circle frame."""
    out = {"n_in_window": int(len(sample))}
    if len(sample) < 10:
        return out
    p = project_sample(R, sample)
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
    ap.add_argument("--param", default=Q_KEY,
                    help="global parameter to sweep (default the halo flattening)")
    ap.add_argument("--values", "--q", nargs="+", type=float, dest="values",
                    default=[0.7, 0.85, 1.0, 1.2, 1.4],
                    help="values of --param to sweep")
    ap.add_argument("--fid", "--q-fid", type=float, default=None, dest="fid",
                    help="value of --param the template is built at; "
                         "default = the config's own value")
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
    ap.add_argument("--noise", action="store_true",
                    help="push every arm through the training observation model (window -> member "
                         "subsample -> Gaia magnitudes -> DR3 errors -> vlos mask) before "
                         "comparing, so sim and real are on the same footing")
    ap.add_argument("--noise-realizations", type=int, default=40,
                    help="independent noise realizations pooled per arm; the chain cuts each one "
                         "to the observed member count, which alone is too sparse to contour")
    ap.add_argument("--noise-subsample", type=int, default=20000,
                    help="particles drawn per noise realization (0 = all); the observation model "
                         "keeps only the observed member count anyway, so this only bounds memory")
    ap.add_argument("--aug", default="stream_global_ibata_grid",
                    help="augmentation preset supplying the observation model")
    ap.add_argument("--outdir", default="data_local/trihedron")
    ap.add_argument("--out", default="trihedron", help="figure filename stem inside --outdir")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--check-fiducial", action="store_true",
                    help="only verify that the remap at the fiducial reproduces the template")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = param_slug(args.param)

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
        stream_dir = outdir / name
        cache = stream_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)

        row_fid = fiducial_row(sim, name)
        v_fid = float(args.fid) if args.fid is not None else row_fid[args.param]
        row_fid[args.param] = v_fid
        print(f"\n=== {name} (j={j})  fiducial {slug}={v_fid:.6g}, "
              f"{args.n_particles} particles ===")

        pot_fid = _host_potential(agama, row_fid, pot_cfg)
        posvel_fid, frame_fid = progenitor_state(agama, pot_fid, row_fid)

        # --- template ---------------------------------------------------------------------------
        xv_fid, m_bound = load_or_run(
            cache / arm_cache_name(name, "rnbody", value_tag(slug, v_fid, v_fid, args), args),
            args.no_cache,
            lambda: run_rnbody(agama, pot_fid, posvel_fid, row_fid, opts,
                               args.n_particles, args.seed, time_unit_gyr),
            label=f"{name} rnbody @ {slug}={v_fid:.6g} (template)",
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
        for q in args.values:
            row = dict(row_fid)
            row[args.param] = float(q)
            pot = _host_potential(agama, row, pot_cfg)
            posvel, frame = progenitor_state(agama, pot, row)

            arms = {"trihedron": remap(agama, pot, posvel, tpl)}
            arms["rnbody"], mb = load_or_run(
                cache / arm_cache_name(name, "rnbody", value_tag(slug, q, v_fid, args), args),
                args.no_cache,
                lambda pot=pot, posvel=posvel, row=row: run_rnbody(
                    agama, pot, posvel, row, opts, args.n_particles, args.seed, time_unit_gyr),
                label=f"{name} rnbody @ {slug}={q:.6g}",
            )
            arms["spray"], _ = load_or_run(
                cache / arm_cache_name(name, "spray", value_tag(slug, q, v_fid, args), args),
                args.no_cache,
                lambda pot=pot, posvel=posvel, row=row: (
                    run_spray(agama, pot, posvel, row, args.n_particles, args.seed,
                              time_unit_gyr), float("nan")),
                label=f"{name} spray  @ {slug}={q:.6g}",
            )

            proj, sample, vmask = {}, {}, {}
            for arm, xv in arms.items():
                proj[arm] = project_xv(np.asarray(xv, dtype=float), frame)
                sample[arm], vmask[arm] = observed_sample(proj[arm], j, args, args.seed)

            m = {arm: arm_metrics(sample[arm], R, edges) for arm in ARM_ORDER}
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
            m.update(halo_m200(agama, row, pot_cfg))
            per_q[f"{q:.6g}"] = m
            e = m["trihedron_vs_rnbody"]
            print(f"  {slug}={q:<9.6g} M200={m['m200_msun']:.3e} c200={m['c200']:.2f}"
                  + "  in-window "
                  + " ".join(f"{a}={m[a].get('n_in_window', 0)}" for a in ARM_ORDER)
                  + f" | per-star |dx| med={e.get('dx_kpc_median', np.nan):.3f}"
                  + f" (along {e.get('dx_along_kpc_median', np.nan):.3f},"
                  + f" perp {e.get('dx_perp_kpc_median', np.nan):.3f}) kpc"
                  + f" | track offset vs truth: trihedron={m['trihedron']['track_offset_deg']:.3f}"
                  + f" spray={m['spray']['track_offset_deg']:.3f} deg"
                  + f" | m_bound={mb:.3g}")
            # kept in memory for the figures, stripped before the json is written
            per_q[f"{q:.6g}"]["_sample"] = sample
            per_q[f"{q:.6g}"]["_vmask"] = vmask

        results[name] = dict(
            j=j, param=args.param, param_fid=v_fid, template=tpl.diagnostics, per_q=per_q,
            R=R.tolist(), real_phi1_range=[float(edges[0]), float(edges[-1])],
        )
        make_figures(name, j, R, real, rvm[rrow][mem], per_q, args, stream_dir, edges, v_fid, slug)
        make_corner_figures(name, R, real, rvm[rrow][mem], per_q, args, stream_dir, slug)

        # metrics json, per stream (drop the in-memory projections)
        res = dict(results[name])
        res["per_q"] = {q: {k: v for k, v in m.items() if not k.startswith("_")}
                        for q, m in res["per_q"].items()}
        path = stream_dir / f"{args.out}_metrics.json"
        path.write_text(json.dumps(res, indent=2))
        print(f"wrote {path}")

    if args.check_fiducial:
        return

    make_error_figure(results, args, outdir, slug)


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


def arm_xy(p, key):
    """One arm's (phi1, quantity), with unmeasured v_los dropped for the v_los panel."""
    sel = p["vlos_mask"] if key == "vlos" else np.ones(len(p["phi1"]), bool)
    return p["phi1"][sel], p[key][sel]


def density_contours(ax, x, y, color, xlim, ylim, levels=(0.68, 0.95), smooth=2.0, label=None):
    """Contours enclosing the given fractions of the sample, from a smoothed 2-D histogram.

    Levels are found by sorting the smoothed density and walking down its cumulative sum, so the
    contour labelled 0.95 encloses 95% of the stars whatever the distribution's shape. The grid is
    sized from the sample: a fixed fine grid leaves ~0.2 stars per cell once the observation model
    has cut each realization to the observed member count, and the contours shatter into confetti.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return
    bins = int(np.clip(np.sqrt(ok.sum() / 4.0), 25, 90))
    H, xe, ye = np.histogram2d(x[ok], y[ok], bins=bins, range=[list(xlim), list(ylim)])
    H = gaussian_filter(H, smooth)
    if H.max() <= 0:
        return
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    lv = sorted({float(flat[min(np.searchsorted(csum, p), len(flat) - 1)]) for p in levels})
    if len(lv) < 1:
        return
    # levels ascend, so the LAST is the innermost (68%) -- draw that solid, the outer 95% dashed
    styles = ["--"] * (len(lv) - 1) + ["-"]
    ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), H.T, levels=lv,
               colors=color, linewidths=1.3, linestyles=styles)
    if label:
        ax.plot([], [], color=color, lw=1.3, label=label)


def make_figures(name, j, R, real, real_vmask, per_q, args, outdir: Path, edges, q_fid,
                 slug) -> None:
    qs = list(per_q.keys())

    # --- phi2 vs phi1, one row per q, all three arms overlaid ---------------------------------
    fig, axes = plt.subplots(len(qs), 1, figsize=(9, 2.3 * len(qs)), sharex=True, sharey=True,
                             squeeze=False)
    lims = []
    proj_cache = {}
    for qi, q in enumerate(qs):
        proj_cache[q] = {}
        for arm in ARM_ORDER:
            s = per_q[q]["_sample"][arm]
            p = project_sample(R, s) if len(s) else None
            if p is not None:
                p["vlos_mask"] = per_q[q]["_vmask"][arm]
                lims.append(p["phi2"])
            proj_cache[q][arm] = p
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
        ax.text(0.01, 0.93, f"{slug} = {q}", transform=ax.transAxes, va="top", fontsize=9)
        if qi == 0:
            ax.legend(loc="upper right", fontsize=7, markerscale=4, framealpha=0.9)
    axes[-1, 0].set_xlabel("phi1 [deg]")
    fig.suptitle(f"{name}: restricted N-body vs trihedron remap vs particle spray "
                 f"(template built at {slug}={q_fid:.6g}, N={args.n_particles})", fontsize=10)
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
                x, y = arm_xy(p, key)
                ax.scatter(x, y, s=1.0, alpha=0.2, c=ARM_COLORS[arm], rasterized=True)
                vals.append(y)
            ax.set_ylim(*robust_lim(*vals))
            if qi == 0:
                ax.set_ylabel(label, fontsize=8)
            if ri == 0:
                ax.set_title(f"{slug} = {q}", fontsize=9)
            ax.tick_params(labelsize=7)
        axes[-1, qi].set_xlabel("phi1 [deg]", fontsize=8)
    fig.suptitle(f"{name}: " + "  ".join(f"{a} ({ARM_COLORS[a]})" for a in ARM_ORDER)
                 + "   black = Gaia members", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.05, hspace=0.08, wspace=0.18)
    path = outdir / f"{args.out}_{name}_overlay.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")

    # --- contour version: the three arms as density contours, real Gaia members as points -----
    # Scatter at 1e5 stars saturates and hides which arm sits where; 68/95% contours make the
    # arms comparable and let the real members read as what they are -- a sparse sample.
    noise_note = (f"noise-convolved, {args.noise_realizations} realizations pooled"
                  if args.noise else "raw simulator output, window cut only")
    fig, axes = plt.subplots(len(QTY), len(qs), figsize=(3.3 * len(qs), 2.2 * len(QTY)),
                             sharex="col", squeeze=False)
    xlim = robust_lim(real["phi1"], pad=0.05)
    # One y-range per observable, shared across q, so the q-trend is readable down each row.
    ylims = {}
    for key, _ in QTY:
        rmask = real_vmask.astype(bool) if key == "vlos" else np.ones(len(real["phi1"]), bool)
        vals = [real[key][rmask]]
        for q in qs:
            for arm in ARM_ORDER:
                p = proj_cache[q][arm]
                if p is not None:
                    vals.append(arm_xy(p, key)[1])
        ylims[key] = robust_lim(*vals)
    for qi, q in enumerate(qs):
        for ri, (key, label) in enumerate(QTY):
            ax = axes[ri, qi]
            rmask = real_vmask.astype(bool) if key == "vlos" else np.ones(len(real["phi1"]), bool)
            ylim = ylims[key]
            for arm in ARM_ORDER:
                p = proj_cache[q][arm]
                if p is None:
                    continue
                x, y = arm_xy(p, key)
                density_contours(ax, x, y, ARM_COLORS[arm], xlim, ylim,
                                 label=arm if (ri == 0 and qi == 0) else None)
            ax.scatter(real["phi1"][rmask], real[key][rmask], s=5, c="k", zorder=5,
                       label="Gaia members" if (ri == 0 and qi == 0) else None)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if qi == 0:
                ax.set_ylabel(label, fontsize=8)
            if ri == 0:
                ax.set_title(f"{slug} = {q}", fontsize=9)
            ax.tick_params(labelsize=7)
        axes[-1, qi].set_xlabel("phi1 [deg]", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="best", framealpha=0.9)
    fig.suptitle(f"{name}: 68/95% density contours per forward model, real Gaia members as points "
                 f"({noise_note})", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.05, hspace=0.10, wspace=0.20)
    path = outdir / f"{args.out}_{name}_contour.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def corner_matrix(p, vmask):
    """The 6 observables as one rectangular array, restricted to stars with a measured v_los.

    ``corner`` needs a rectangle, and v_los is missing for most stars once the observation model has
    run, so the corner is by construction the measured-v_los subset -- stated in the suptitle.
    """
    sel = np.asarray(vmask, bool)
    cols = [p["phi1"], p["phi2"], p["plx"], p["mu_phi1"], p["mu_phi2"], p["vlos"]]
    X = np.column_stack([np.asarray(c, float)[sel] for c in cols])
    return X[np.isfinite(X).all(axis=1)]


CORNER_LABELS = ["phi1 [deg]", "phi2 [deg]", "parallax [mas]",
                 "mu_phi1 [mas/yr]", "mu_phi2 [mas/yr]", "vlos [km/s]"]


def make_corner_figures(name, R, real, real_vmask, per_q, args, outdir: Path, slug) -> None:
    """One observable-space corner per swept value: the three arms as contours, real data as points.

    The vs-phi1 panels show each observable's marginal track; this shows the CORRELATIONS between
    them, which is where a surrogate forward model can agree marginally and still be wrong.
    """
    import corner
    from matplotlib.lines import Line2D

    real_X = corner_matrix(real, real_vmask)
    noise_note = (f"noise-convolved, {args.noise_realizations} realizations pooled"
                  if args.noise else "raw simulator output, window cut only")
    for q in per_q:
        data = {}
        for arm in ARM_ORDER:
            s = per_q[q]["_sample"][arm]
            if len(s) == 0:
                continue
            p = project_sample(R, s)
            X = corner_matrix(p, per_q[q]["_vmask"][arm])
            if len(X) >= 50:
                data[arm] = X
        if not data:
            continue
        stack = list(data.values()) + [real_X]
        ranges = [robust_lim(*[X[:, i] for X in stack], pad=0.10) for i in range(len(CORNER_LABELS))]

        # Bin count from the sample, as in density_contours: the observation model cuts each
        # realization to the observed member count, and a fixed fine grid shatters the contours.
        nmin = min(len(X) for X in data.values())
        bins = int(np.clip(np.sqrt(nmin / 4.0), 20, 50))

        fig = None
        for arm, X in data.items():
            fig = corner.corner(
                X, fig=fig, labels=CORNER_LABELS, range=ranges, color=ARM_COLORS[arm],
                bins=bins, plot_datapoints=False, plot_density=False, fill_contours=False,
                smooth=1.0, levels=(0.68, 0.95), hist_kwargs={"density": True},
                label_kwargs={"fontsize": 9},
            )
        if len(real_X):
            # overplot_points already forces linestyle="none"; passing ls too is a matplotlib error
            corner.overplot_points(fig, real_X, color="k", marker=".", ms=2.5)
        handles = [Line2D([], [], color=ARM_COLORS[a], lw=1.5, label=a) for a in data]
        handles.append(Line2D([], [], color="k", marker=".", ls="none", label="Gaia members"))
        fig.legend(handles=handles, loc="upper right", fontsize=10, frameon=False)
        fig.suptitle(f"{name}  {slug} = {q}   68/95% contours, measured-v_los stars only "
                     f"({noise_note})", fontsize=11, y=1.0)
        path = outdir / f"{args.out}_{name}_corner_{slug}{q}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


def make_error_figure(results, args, outdir: Path, slug) -> None:
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
    axes[0].set_xlabel(slug)
    axes[0].set_ylabel("median |d phi2 track| vs rnbody [deg]")
    axes[0].set_title("accuracy vs ground truth (distributional)")
    axes[1].set_xlabel(slug)
    axes[1].set_ylabel("per-star |dx| vs rnbody [kpc]")
    axes[1].set_title("per-star displacement\n(along-track = star decorrelation, not remap error)")
    for ax in axes:
        for name, res in results.items():
            ax.axvline(res["param_fid"], color="0.5", ls=":", lw=1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / f"{args.out}_error_vs_param.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
