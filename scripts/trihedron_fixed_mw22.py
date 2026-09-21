"""Trihedron (Frenet-Serret) remap twin of the fixed-MW2022 gala spray realizations.

Same controlled setup as `stream_gala_spray_mw22_fixed`: ONE Galaxy (MW2022 fiducial), fixed
progenitors, the only variation being the progenitors' present-day phase-space coordinates within
their measurement errors. Here the stream is simulated ONCE per stream with gala's Chen+2024 spray at
the prior-MEAN phase space (the template), frozen into the trihedron frame of that fiducial orbit
(scripts/trihedron.py), and for every stored phase-space draw of the spray dataset only the
progenitor orbit is re-integrated and the stored offsets dropped onto the new trihedra. The output
npz has the same grouped layout as the spray set (row i uses the SAME draw as spray row i), so the
two can be compared row for row and the plot/table tooling runs unchanged.

The orbit integration + trihedra use an AGAMA copy of the gala potential (three Miyamoto-Nagai
disks read off gala's MN3, two Hernquist spheroids, spherical NFW); its circular velocity is checked
against gala before use. Sky projection = the astropy default frame, as in stream_gala.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trihedron import build_template, remap  # noqa: E402

NAMES = {0: "Pal5", 1: "NGC3201", 2: "M68"}
LOCAL_KEYS = ("t_end", "m_progenitor", "a_progenitor", "ra", "dec", "vr", "r", "mu_ra_cosdec", "mu_dec", "m_progenitor_final")


def gala_fiducial(cfg_name):
    """Globals + per-stream local means from the fixed simulator config (all identity / prior means)."""
    from hydra import compose, initialize_config_dir
    from hydrabflow.config import register_configs
    register_configs()
    with initialize_config_dir(config_dir=os.path.abspath("conf"), version_base=None):
        cfg = compose("config", overrides=[f"simulator={cfg_name}"])
    P = cfg.simulator.params
    glob = {k: float(v.prior_parameters[0]) for k, v in P.priors_global.items()}
    loc = {s: {k: float(v.prior_parameters[0]) for k, v in d.items()} for s, d in P.priors_local.items()}
    opts = dict(n_steps=int(P.n_steps), n_particles_per_release=int(P.n_particles_per_release),
                q_ref_r_kpc=float(P.q_ref_r_kpc), c_phi_bracket=tuple(P.c_phi_bracket),
                mass_loss=P.mass_loss, integrator=P.integrator, raise_errors=True)
    return glob, loc, opts, int(P.n_particles)


def agama_potential(agama, glob, rho_c):
    """AGAMA copy of stream_gala._build_potential (spherical halo only: q_rho=1 here)."""
    import astropy.units as u
    import gala.potential as gp
    from gala.units import galactic
    from hydrabflow.simulators.stream_gala import nfw_m_rs_from_m200_c
    assert abs(glob["q_rho_halo"] - 1.0) < 1e-12, "agama twin implemented for a spherical halo only"
    m_nfw, r_s = nfw_m_rs_from_m200_c(10.0 ** glob["log10_M200_halo"], glob["c200_halo"], rho_c)
    mn = gp.MN3ExponentialDiskPotential(m=glob["m_disk"] * u.Msun, h_R=glob["h_R_disk"] * u.kpc,
                                        h_z=glob["h_z_disk"] * u.kpc, units=galactic).get_three_potentials()
    comps = [dict(type="MiyamotoNagai", mass=float(c.parameters["m"].to(u.Msun).value),
                  scaleRadius=float(c.parameters["a"].to(u.kpc).value),
                  scaleHeight=float(c.parameters["b"].to(u.kpc).value)) for c in mn.values()]
    for m, c in ((glob["m_bulge"], glob["c_bulge"]), (glob["m_nucleus"], glob["c_nucleus"])):
        comps.append(dict(type="Spheroid", gamma=1, beta=4, alpha=1, scaleRadius=c, mass=m))
    comps.append(dict(type="Spheroid", gamma=1, beta=3, alpha=1, scaleRadius=r_s,
                      densityNorm=m_nfw / (4 * np.pi * r_s**3)))
    return agama.Potential(*[agama.Potential(**c) for c in comps])


def posvel_from_icrs(row):
    import astropy.coordinates as coord
    import astropy.units as u
    c = coord.SkyCoord(ra=row["ra"] * u.deg, dec=row["dec"] * u.deg, distance=row["r"] * u.kpc,
                       pm_ra_cosdec=row["mu_ra_cosdec"] * u.mas / u.yr, pm_dec=row["mu_dec"] * u.mas / u.yr,
                       radial_velocity=row["vr"] * u.km / u.s, frame="icrs").transform_to(coord.Galactocentric())
    return np.array([c.x.to(u.kpc).value, c.y.to(u.kpc).value, c.z.to(u.kpc).value,
                     c.v_x.to(u.km / u.s).value, c.v_y.to(u.km / u.s).value, c.v_z.to(u.km / u.s).value])


def project_xv(xv):
    """sky_projection is all-or-nothing on NaN rows (2026-08-28 bug): project the finite subset."""
    from hydrabflow.simulators.stream_common import sky_projection
    out = np.full_like(xv, np.nan)
    ok = np.isfinite(xv).all(1)
    if ok.any():
        out[ok] = sky_projection(xv[ok][None])[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spray", required=True, help="fixed-potential spray npz whose stored draws are reused")
    ap.add_argument("--out", required=True, help="output npz (grouped layout, raw Galactocentric->ICRS, no window cap)")
    ap.add_argument("--simulator", default="stream_gala_spray_mw22_fixed")
    ap.add_argument("--T-myr", type=float, default=200.0)
    ap.add_argument("--n-knots", type=int, default=2001)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    import agama
    from hydrabflow.simulators.stream_gala import _simulate_one_gala, RHO_CRIT_PLANCK18_MSUN_KPC3, _build_potential
    agama.setUnits(length=1, velocity=1, mass=1)
    tu = agama.getUnits()["time"]
    time_unit_gyr = float(getattr(tu, "value", tu)) / 1e3
    T = args.T_myr * 1e-3 / time_unit_gyr

    glob, loc, opts, n_particles = gala_fiducial(args.simulator)
    opts["rho_c"] = RHO_CRIT_PLANCK18_MSUN_KPC3
    pot = agama_potential(agama, glob, opts["rho_c"])
    # --- check the agama twin against gala on the rotation curve ---------------------------------
    import astropy.units as u
    import gala.potential as gp
    from gala.units import galactic
    gpot, _ = _build_potential(gp, u, galactic, glob, opts)
    r = np.array([1, 2, 4, 8.178, 12, 20, 30, 50.0])
    q = np.zeros((3, len(r)))
    q[0] = r
    vg = gpot.circular_velocity(q * u.kpc).to(u.km / u.s).value
    xyz = np.column_stack([r, 0 * r, 0 * r])
    f = pot.force(xyz)
    va = np.sqrt(-r * f[:, 0])
    dev = np.abs(va / vg - 1).max()
    print(f"agama twin vs gala v_circ: max |frac dev| = {dev:.2e} over r={r.tolist()} kpc")
    assert dev < 1e-3, "agama potential does not match gala"

    d = np.load(args.spray)
    j = np.asarray(d["j"]).reshape(d["j"].shape[:2]).astype(int)
    n_rows, n_streams = j.shape
    out = np.full((n_rows, n_streams, n_particles, 6), np.nan, dtype=np.float32)
    report = {"T_myr": args.T_myr, "n_knots": args.n_knots, "vcirc_max_frac_dev": float(dev), "streams": {}}
    obs_r = np.array([8.178])
    for col in range(n_streams):
        nm = NAMES[int(j[0, col])]
        row = dict(glob)
        row.update(loc[nm])
        t0 = time.time()
        xv_fid, _, _, _, _, _ = _simulate_one_gala(row, n_particles, obs_r, args.seed, opts)
        t_spray = time.time() - t0
        pv_fid = posvel_from_icrs(row)
        t0 = time.time()
        tpl = build_template(agama, pot, pv_fid, xv_fid, T, args.n_knots)
        t_tpl = time.time() - t0
        rt = remap(agama, pot, pv_fid, tpl)
        rt_err = float(np.nanmax(np.abs(rt - xv_fid)))
        t0 = time.time()
        for i in range(n_rows):
            rr = dict(row)
            rr.update({k: float(d[k][i, col, 0]) for k in ("vr", "r", "mu_ra_cosdec", "mu_dec")})
            out[i, col] = project_xv(remap(agama, pot, posvel_from_icrs(rr), tpl))
        t_remap = (time.time() - t0) / n_rows
        diag = {k: (float(v) if np.isscalar(v) else v) for k, v in tpl.diagnostics.items()}
        report["streams"][nm] = dict(template_diag=diag, fiducial_roundtrip_max_abs=rt_err,
                                     t_spray_s=t_spray, t_template_s=t_tpl, t_remap_per_row_s=t_remap)
        print(f"{nm}: spray {t_spray:.1f}s, template {t_tpl:.1f}s, remap {t_remap*1e3:.0f} ms/row, "
              f"round-trip {rt_err:.1e}, boundary {diag.get('boundary_frac', float('nan')):.3f}, "
              f"ambiguous {diag.get('ambiguous_frac', float('nan')):.3f}")
    arrays = {"sim_data_projected": out, "j": d["j"]}
    for k in LOCAL_KEYS:
        if k in d.files:
            arrays[k] = d[k]
    np.savez_compressed(args.out, **arrays)
    json.dump(report, open(os.path.splitext(args.out)[0] + "_report.json", "w"), indent=1, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
