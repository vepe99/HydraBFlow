#!/usr/bin/env python
"""Freeze a finished stream simulation into the Frenet-Serret trihedron of its progenitor's orbit.

Input: a grouped npz from ``simulate_multistream`` that stored ``sim_data_carthesian`` (the full
Galactocentric cloud), together with its ``.hydra`` config snapshot. Output: one template per
stream, which ``stream_trihedron`` then remaps into any new potential in milliseconds.

The template MUST be built in the potential the stars actually moved in — reconstructed here from
the run's own resolved config, not from whatever family the training set will use — otherwise every
star's ``t_hat`` label is attached to the wrong orbit.

    .venv/bin/python scripts/build_trihedron_template.py \
        --sim data_local/mcmillan17_mean_tend_best/mcmillan17_mean.npz \
        --out data_local/mcmillan17_mean_tend_best/trihedron_template.npz \
        --T-gyr 0.6

``--compare-simulator`` additionally reports how far the fiducial potential sits from the prior the
template will be used over (step 6 of the plan): the remap is exact at its own fiducial and degrades
away from it, so that distance is the headline caveat of any dataset built this way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from hydrabflow.simulators.stream_agama import (  # noqa: E402
    _agama,
    _halo_params_m200c,
    _host_potential,
    _resolve_pot_cfg,
    _solar_frame,
)
from hydrabflow.simulators.trihedron import auto_window, build_template, remap  # noqa: E402

NAMES = {0: "Pal5", 1: "NGC3201", 2: "M68"}
GLOBAL_SCALARS = "ndim2"  # dataset keys shaped (n, 1) are globals; (n, s, 1) are locals
LOCAL_KEYS = (
    "m_progenitor", "a_progenitor", "t_end",
    "ra", "dec", "vr", "r", "mu_ra_cosdec", "mu_dec",
)
FROZEN_LOCAL_KEYS = ("m_progenitor", "a_progenitor", "t_end")


def pot_cfg_from_snapshot(sim_path: Path) -> dict:
    """The host-potential configuration of the run that produced ``sim_path``.

    Read from its ``<stem>.hydra/config.yaml`` rather than re-composed from a config NAME, so the
    template records the potential that was actually integrated even if the yaml is edited later.
    """
    from omegaconf import OmegaConf

    snap = sim_path.with_suffix("") .parent / f"{sim_path.stem}.hydra" / "config.yaml"
    if not snap.exists():
        raise FileNotFoundError(f"no config snapshot at {snap}")
    params = OmegaConf.to_container(OmegaConf.load(snap).simulator.params, resolve=True)
    # Same defaults `AgamaStreamSimulator._pot_cfg` applies, so an absent key means "legacy value".
    cfg = _resolve_pot_cfg(None)
    for key in cfg:
        if key in params and params[key] is not None:
            cfg[key] = params[key]
    cfg["halo_r_t_kpc"] = float(cfg["halo_r_t_kpc"])
    return cfg


def row_of(data, j: int) -> dict:
    """Globals + stream ``j``'s locals of the (single-row) fiducial dataset, as the worker saw them."""
    row = {}
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape[1] == 1:          # global (n, 1)
            row[key] = float(arr[0, 0])
        elif arr.ndim == 3 and arr.shape[2] == 1:        # local (n, s, 1)
            row[key] = float(arr[0, j, 0])
    return row


def progenitor_state(agama, pot, row: dict):
    """Present-day Galactocentric 6D state of the progenitor, and the row's solar frame.

    Verbatim ``stream_agama._simulate_one``: the observed ICRS coordinates are fixed, but the frame
    they are converted through depends on the potential.
    """
    frame = _solar_frame(agama, pot, row)
    l0, b0, pml0, pmb0 = agama.transformCelestialCoords(
        agama.fromICRStoGalactic,
        row["ra"] * np.pi / 180, row["dec"] * np.pi / 180,
        row["mu_ra_cosdec"], row["mu_dec"],
    )
    return np.array(
        agama.getGalactocentricFromGalactic(
            l0, b0, row["r"], pml0 * 4.74, pmb0 * 4.74, row["vr"],
            galcen_distance=frame[0], galcen_v_sun=frame[1:4], z_sun=frame[4],
        )
    ), frame


def halo_in_m200c(agama, row: dict, pot_cfg: dict) -> dict:
    """Report a ``rho_a`` halo in the (M200, c') coordinates the training prior uses.

    Inverts ``_halo_params_m200c`` by bisection on log10 M200 at fixed shape: for each trial mass the
    forward map gives a (densityNorm, scaleRadius) pair, and we match the scale radius, then read off
    the concentration. Purely diagnostic — it says where the template's fiducial sits inside the
    prior it will be used over.
    """
    if str(pot_cfg["halo_parameterization"]) == "m200_c":
        return {"log10_M200": row["log10_M200_TwoPowerTriaxial_halo"],
                "ln_cvprime": row["ln_cvprime_TwoPowerTriaxial_halo"]}
    target_a = row["a_TwoPowerTriaxial_halo"]
    gamma = row["gamma_TwoPowerTriaxial_halo"]

    def scale_radius(log10_m200: float, ln_c: float) -> float:
        p = dict(row, log10_M200_TwoPowerTriaxial_halo=log10_m200,
                 ln_cvprime_TwoPowerTriaxial_halo=ln_c,
                 gamma_TwoPowerTriaxial_halo=gamma)
        return float(_halo_params_m200c(agama, p, pot_cfg)["scaleRadius"])

    # Solve the 2-D (mass, concentration) match as nested bisections: concentration sets the ratio
    # r200/r_h, mass sets r200, so for each mass there is one concentration reproducing BOTH the
    # scale radius and the density norm.
    best = None
    for log10_m200 in np.linspace(11.0, 13.0, 201):
        lo, hi = 1.0, 4.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if scale_radius(log10_m200, mid) > target_a:
                lo = mid
            else:
                hi = mid
        ln_c = 0.5 * (lo + hi)
        p = dict(row, log10_M200_TwoPowerTriaxial_halo=log10_m200,
                 ln_cvprime_TwoPowerTriaxial_halo=ln_c)
        h = _halo_params_m200c(agama, p, pot_cfg)
        err = abs(h["densityNorm"] / row["rho_TwoPowerTriaxial_halo"] - 1.0)
        if best is None or err < best[0]:
            best = (err, log10_m200, ln_c, float(h["scaleRadius"]), float(h["densityNorm"]))
    err, log10_m200, ln_c, a_fit, rho_fit = best
    return {"log10_M200": log10_m200, "ln_cvprime": ln_c, "c_vprime": float(np.exp(ln_c)),
            "M200_msun": float(10.0 ** log10_m200), "scaleRadius_fit": a_fit,
            "densityNorm_fit": rho_fit, "densityNorm_rel_err": err}


def vcirc_offset(agama, row: dict, pot_cfg: dict, simulator: str) -> dict | None:
    """Fractional ``v_circ(r)`` difference, fiducial potential vs the prior centre of ``simulator``.

    This is the "how far outside the training family is the template's exact point" number. It is
    NOT a correction — nothing uses it — but a dataset built on a distant fiducial should say so.
    """
    try:
        from compare_trihedron_vs_spray import fiducial_row, load_simulator
    except Exception:
        return None
    sim = load_simulator(simulator)
    centre = fiducial_row(sim, next(iter(sim.target_streams)))
    pot_centre = _host_potential(agama, centre, sim._pot_cfg)
    pot_fid = _host_potential(agama, row, pot_cfg)
    r = np.geomspace(2.0, 60.0, 60)
    xyz = np.column_stack([r, np.zeros_like(r), np.zeros_like(r)])
    vc = lambda pot: np.sqrt(-r * pot.force(xyz)[:, 0])  # noqa: E731
    frac = np.abs(vc(pot_fid) / vc(pot_centre) - 1.0)
    return {"simulator": simulator, "r_kpc": [2.0, 60.0],
            "median_abs_frac_dev": float(np.median(frac)),
            "max_abs_frac_dev": float(np.max(frac))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True, help="fiducial grouped npz with sim_data_carthesian")
    ap.add_argument("--out", required=True, help="output template npz")
    ap.add_argument("--T-gyr", type=float, default=0.6,
                    help="half-width of the orbit window the trihedron is built on [Gyr]")
    ap.add_argument("--knot-myr", type=float, default=0.2, help="orbit knot spacing [Myr]")
    ap.add_argument("--velocity", choices=("relative", "absolute"), default="relative")
    ap.add_argument("--auto-T", action="store_true",
                    help="grow T until boundary_frac < 1e-3 instead of using --T-gyr as given")
    ap.add_argument("--row", type=int, default=0, help="dataset row to freeze")
    ap.add_argument("--compare-simulator", default="stream_agama_rnbody_ibata_m200c_v4",
                    help="report the v_circ offset to this config's prior centre ('' to skip)")
    ap.add_argument("--fig", default="", help="diagnostic figure path (default: alongside --out)")
    args = ap.parse_args()

    sim_path = Path(args.sim).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agama = _agama()
    tu = agama.getUnits()["time"]
    time_unit_gyr = float(getattr(tu, "value", tu)) / 1e3
    T = args.T_gyr / time_unit_gyr
    n_knots = int(round(2 * args.T_gyr * 1e3 / args.knot_myr)) + 1

    data = np.load(sim_path)
    if "sim_data_carthesian" not in data.files:
        raise KeyError(
            f"{sim_path} has no 'sim_data_carthesian' — the template needs the full Galactocentric "
            "cloud, so the fiducial run must NOT use store_window_subsample"
        )
    xv_all = np.asarray(data["sim_data_carthesian"], dtype=float)[args.row]  # (s, P, 6)
    j_all = np.asarray(data["j"], dtype=float)[args.row].reshape(-1).astype(int)
    pot_cfg = pot_cfg_from_snapshot(sim_path)

    print(f"fiducial: {sim_path}")
    print(f"  potential cfg: {pot_cfg}")
    print(f"  window T = {args.T_gyr} Gyr ({T:.1f} agama), {n_knots} knots "
          f"({args.knot_myr} Myr spacing)\n")

    store: dict[str, np.ndarray] = {}
    report: dict = {"fiducial": str(sim_path), "T_gyr": args.T_gyr, "n_knots": n_knots,
                    "velocity": args.velocity, "pot_cfg": {k: str(v) for k, v in pot_cfg.items()},
                    "streams": {}}
    t_knots = None
    for s, j in enumerate(j_all):
        name = NAMES.get(j, str(j))
        row = row_of(data, s)
        pot_fid = _host_potential(agama, row, pot_cfg)
        posvel, frame = progenitor_state(agama, pot_fid, row)
        xv = xv_all[s]

        if args.auto_T:
            tpl, T_used, knots_used = auto_window(
                agama, pot_fid, posvel, xv, T, n_knots,
                t_max=row["t_end"] / time_unit_gyr, velocity=args.velocity)
        else:
            tpl = build_template(agama, pot_fid, posvel, xv, T, n_knots, velocity=args.velocity)
            T_used, knots_used = T, n_knots

        # The one check that says the template is a faithful description of the simulation it came
        # from: remapping into the SAME potential must return the very cloud it was built from.
        xv_back = remap(agama, pot_fid, posvel, tpl)
        ok = np.isfinite(xv).all(axis=1)
        err = float(np.nanmax(np.abs(xv_back[ok, 0:3] - xv[ok, 0:3])))
        if not err < 1e-6:
            raise AssertionError(f"{name}: remap is not exact at the fiducial ({err:.3e} kpc)")

        d = tpl.diagnostics
        print(f"{name:9s} P={tpl.n_stars:6d}  boundary={d['boundary_frac']:.4f}  "
              f"ambiguous={d['ambiguous_frac']:.4f}  degenerate={d['degenerate_frame_frac']:.2e}  "
              f"nan={d['nan_star_frac']:.4f}  |remap-fid|={err:.2e} kpc")
        if d["boundary_frac"] > 1e-3:
            print(f"  WARNING {name}: {d['boundary_frac']:.1%} of stars pin to the window edge — "
                  f"their nearest orbit point lies outside +-{args.T_gyr} Gyr. Rerun with --auto-T.")

        store[f"t_hat_{j}"] = tpl.t_hat
        store[f"c_{j}"] = tpl.c
        store[f"w_{j}"] = tpl.w
        store[f"posvel_{j}"] = posvel
        store[f"frame_{j}"] = np.asarray(frame, dtype=float)
        for k in FROZEN_LOCAL_KEYS:
            store[f"fiducial_{k}_{j}"] = np.float64(row[k])
        t_knots = tpl.t_knots
        report["streams"][name] = {
            "j": int(j), "n_stars": tpl.n_stars, "T_gyr": float(T_used * time_unit_gyr),
            "n_knots": int(knots_used), "remap_fiducial_max_dx_kpc": err,
            "frozen_locals": {k: row[k] for k in FROZEN_LOCAL_KEYS},
            **{k: float(v) for k, v in d.items()},
        }

    store["t_knots"] = t_knots
    store["streams"] = np.asarray(sorted(j_all), dtype=np.int64)
    store["velocity"] = np.asarray(args.velocity)
    np.savez_compressed(out_path, **store)
    print(f"\nsaved template -> {out_path}")

    # Where does the fiducial sit relative to the prior the template will be used over?
    row0 = row_of(data, 0)
    try:
        report["halo_in_m200c"] = halo_in_m200c(agama, row0, pot_cfg)
    except Exception as exc:  # diagnostic only, never fatal
        report["halo_in_m200c"] = {"error": str(exc)}
    if args.compare_simulator:
        try:
            report["vcirc_offset_to_prior_centre"] = vcirc_offset(
                agama, row0, pot_cfg, args.compare_simulator)
        except Exception as exc:
            report["vcirc_offset_to_prior_centre"] = {"error": str(exc)}
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"saved report   -> {out_path.with_suffix('.json')}")
    if "halo_in_m200c" in report and "log10_M200" in report["halo_in_m200c"]:
        h = report["halo_in_m200c"]
        print(f"  fiducial halo in m200_c coords: log10 M200 = {h['log10_M200']:.3f}, "
              f"ln c' = {h['ln_cvprime']:.3f}")
    off = report.get("vcirc_offset_to_prior_centre")
    if off and "median_abs_frac_dev" in off:
        print(f"  v_circ vs {off['simulator']} prior centre over 2-60 kpc: "
              f"median {off['median_abs_frac_dev']:.1%}, max {off['max_abs_frac_dev']:.1%}")

    make_figure(store, j_all, report, Path(args.fig) if args.fig
                else out_path.with_name(out_path.stem + "_diagnostics.png"))


def make_figure(store, j_all, report, path: Path) -> None:
    """t_hat distribution per stream: where along the orbit window the stream actually lives."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(j_all), figsize=(4.2 * len(j_all), 3.2), squeeze=False)
    for ax, j in zip(axes[0], j_all):
        name = NAMES.get(j, str(j))
        t_hat = store[f"t_hat_{j}"]
        t_knots = store["t_knots"]
        ax.hist(t_hat[np.isfinite(t_hat)], bins=80, color="#2a78d6", alpha=0.85)
        ax.axvline(t_knots[0], color="#eb6834", ls="--", lw=1)
        ax.axvline(t_knots[-1], color="#eb6834", ls="--", lw=1)
        d = report["streams"][name]
        ax.set_title(f"{name}  boundary {d['boundary_frac']:.3f}  ambig {d['ambiguous_frac']:.3f}",
                     fontsize=9)
        ax.set_xlabel("t_hat [agama time units]")
        ax.set_ylabel("stars")
    fig.suptitle("trihedron template: nearest-orbit-point time per star "
                 "(dashed = window edges)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved figure   -> {path}")


if __name__ == "__main__":
    main()
