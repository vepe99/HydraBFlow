#!/usr/bin/env python
"""Pal 5 x Milky-Way potential: nested sampling with a galax Chen+2024 particle-spray stream.

ONE file, four modes (run with the dedicated env: ``.venv-galax/bin/python scripts/pal5_galax_ns.py``):

  bench    time / sanity-check the forward model and the likelihood on one free GPU
  prior    prior predictive checks (do the priors generate streams like the observed one?)
  run      launch one independent nested-sampling run per free GPU (autocvd picks them) and wait
  worker   (internal) a single nested-sampling run on the GPU it was handed
  post     merge the runs -> evidence, posterior, corner plot, posterior predictive checks

Forward model (all galax):
  * host   = Cautun+2020 thin + thick exponential stellar disks (``MN3ExponentialPotential``) and bulge
             (its Cautun mass in a ``PlummerPotential`` of the same half-mass radius: Pal 5 never comes
             inside 5 kpc, where only the monopole matters), all FIXED, + a density-flattened NFW halo
             (``TriaxialNFWPotential`` with q1 = 1, q2 = q) with free scale mass, scale radius, flattening.
  * stream = Chen, Gnedin & Li (2024) release recipe (``galax.dynamics.ChenStreamDF``) along the progenitor
             orbit, particles then integrated in host + the progenitor's own moving Plummer potential
             (``TranslatedPotential(PlummerPotential, TimeDependentTranslationParameter)``).
             galax 0.1.0's ``MockStreamGenerator`` cannot evaluate a progenitor potential, so the generator
             is assembled here from those galax pieces (validated against gala, see ``bench --gala``).
  * N      = 2 x 5000 release times = 10^4 particles, fixed random key (common random numbers), so the
             likelihood is a deterministic function of the parameters.

Likelihood: the mock stream is binned in the stream longitude phi1; in every bin the model gives a local
linear track and an intrinsic dispersion for (phi2, parallax, pm_ra*, pm_dec, v_los). Each observed star is
Gaussian about its bin's track with variance = model dispersion^2 (x (1 + 1/n_bin)) + Gaia error^2 +
sigma_inflate^2, the five ``sigma_inflate`` (one per observable) being sampled (they absorb what a particle spray cannot
reproduce of a full N-body stream), mixed with a uniform outlier component of sampled fraction.

``--likelihood bin``: observed bin centre vs model bin centre (see ``make_loglike``).
``--likelihood bspline``: the density of the B-spline TRACKS of the observed stars (``make_loglike_bspline``):
the same Gaussian stars without the outlier term, pushed through the fixed linear core080 spline operators
(``bspline_ops.npz`` of ``scripts/pal5_sbi_comparison.py``, ``--noise nsbspline``).
``--rotation-curve jiao | eilers19``: adds the Gaussian log-likelihood of an observed rotation curve (``make_loglike_rc``) to
whichever stream likelihood is chosen.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz"
GAIA_ERR = REPO / "assets/gaia/gaia_DR3_erorr_6D.txt"
OUT = REPO / "outputs/pal5_galax_ns"


def pick_free_gpus(n: int) -> list[str]:
    """Free GPUs via autocvd (waits until ``n`` are free). Must run before jax is imported."""
    from autocvd import autocvd

    autocvd(num_gpus=n)
    return os.environ["CUDA_VISIBLE_DEVICES"].split(",")


if __name__ == "__main__":
    _mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if _mode in ("bench", "post", "prior") and "--cpu" not in sys.argv:
        if os.environ.get("PAL5_GPU"):  # a GPU chosen by hand (index or UUID) where autocvd cannot run
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["PAL5_GPU"]
        else:
            pick_free_gpus(1)
    elif _mode in ("run", "selftest") or "--cpu" in sys.argv:
        os.environ["JAX_PLATFORMS"] = "cpu"  # the launcher itself never touches a GPU
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import warnings

warnings.filterwarnings("ignore")

import jax

# Precision: float64. float32 (what StreaMAX uses; PAL5_X64=0) is ~5x faster on these GPUs but its
# round-off error does not converge with the step count (dlogL up to 7 vs the reference), so it is off.
X64 = os.environ.get("PAL5_X64", "1") == "1"
jax.config.update("jax_enable_x64", X64)

import diffrax as dfx
import interpax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import galax.dynamics as gd
import galax.potential as gp
import unxt as u

# ======================================================================================================
# Constants
# ======================================================================================================
USYS = "galactic"  # kpc, Myr, Msun
KMS = 977.7922216807891  # kpc/Myr -> km/s
K_PM = 4.740470463533348  # km/s per (mas/yr * kpc)

# Cautun+2020 (MNRAS 494, 4291; table 2 = agama's Cautun20.ini) baryons, held fixed.
THIN = dict(sigma0=7.31e8, Rd=2.63, zd=0.30)  # Msun/kpc^2, kpc, kpc
THICK = dict(sigma0=1.01e8, Rd=3.80, zd=0.90)
BULGE = dict(rho0=1.03e11, r0=0.075, rcut=2.1, alpha=1.8, q=0.5)  # Msun/kpc^3, kpc
M_THIN = 2 * np.pi * THIN["sigma0"] * THIN["Rd"] ** 2
M_THICK = 2 * np.pi * THICK["sigma0"] * THICK["Rd"] ** 2


def _bulge_mass(r_max: float = np.inf) -> float:
    """Mass of the Cautun/McMillan bulge rho0 (1 + m/r0)^-alpha exp(-(m/rcut)^2), m^2 = R^2 + (z/q)^2,
    inside the ellipsoid m < r_max (total by default)."""
    from scipy.integrate import quad

    b = BULGE
    f = lambda m: m**2 * (1 + m / b["r0"]) ** (-b["alpha"]) * np.exp(-((m / b["rcut"]) ** 2))
    return float(4 * np.pi * b["q"] * b["rho0"] * quad(f, 0, min(r_max, 30 * b["rcut"]), limit=400)[0])


def _bulge_plummer_scale() -> float:
    """Plummer scale with the same half-mass (ellipsoidal) radius as the Cautun bulge."""
    from scipy.optimize import brentq

    m_half = brentq(lambda m: _bulge_mass(m) - 0.5 * M_BULGE, 1e-3, 20.0)
    return float(m_half / 1.3048)


M_BULGE = _bulge_mass()
B_BULGE = _bulge_plummer_scale()

# Pal 5 present-day position (fixed) -- the project's values (conf/simulator/stream_agama.yaml).
PAL5_RA, PAL5_DEC = 229.022, -0.112

LOG10_M_NOW = float(np.log10(1.34e4))  # present-day Pal 5 mass, Baumgardt & Hilker (2018), fixed

N_RELEASE = 5000  # release times; two arms -> 10^4 stream particles
N_STEPS = 2000  # KDK leapfrog steps per particle, release -> today (see `bench`)
GL_ORDER = 10  # Gauss-Legendre nodes of the flattened-NFW integral (force error < 4e-7 for q in [0.5, 1.5])
STREAM_KEY = 20261119  # fixed particle-spray random key

# ---- parameters: (name, kind, a, b, latex) ; kind 'u' = Uniform[a, b], 'n' = Normal(a, b) ------------
PARAMS = [
    ("log10_m_nfw", "u", 11.0, 12.5, r"$\log_{10} M_s/M_\odot$"),  # NFW scale mass 4 pi rho0 r_s^3
    ("r_s", "u", 5.0, 40.0, r"$r_s$ [kpc]"),
    ("q", "u", 0.5, 1.5, r"$q_z$"),  # density flattening (z/x axis ratio of the halo)
    ("dist", "n", 20.6, 0.2, r"$d$ [kpc]"),  # RR Lyrae distance, Price-Whelan+2019
    ("pmra", "n", -2.736, 0.064, r"$\mu_{\alpha*}$ [mas/yr]"),
    ("pmdec", "n", -2.646, 0.064, r"$\mu_\delta$ [mas/yr]"),
    ("vr", "n", -58.6, 0.21, r"$v_r$ [km/s]"),
    ("log10_m_prog", "u", LOG10_M_NOW, 5.7, r"$\log_{10} M_{\rm prog,0}/M_\odot$"),  # INITIAL mass
    ("a_prog", "u", 10.0, 40.0, r"$a_{\rm prog}$ [pc]"),  # Plummer scale radius
    ("t_end", "u", 1.0, 5.0, r"$t_{\rm strip}$ [Gyr]"),
    ("sig_phi2", "u", 0.0, 1.0, r"$\sigma^{\rm inf}_{\phi_2}$ [deg]"),
    ("sig_plx", "u", 0.0, 1.0, r"$\sigma^{\rm inf}_{\varpi}$ [mas]"),
    ("sig_pmra", "u", 0.0, 0.5, r"$\sigma^{\rm inf}_{\mu_{\alpha*}}$ [mas/yr]"),
    ("sig_pmdec", "u", 0.0, 0.5, r"$\sigma^{\rm inf}_{\mu_\delta}$ [mas/yr]"),
    ("sig_vlos", "u", 0.0, 15.0, r"$\sigma^{\rm inf}_{v}$ [km/s]"),
    ("f_out", "u", 0.0, 0.2, r"$f_{\rm out}$"),
]
NAMES = [p[0] for p in PARAMS]
NDIM = len(PARAMS)
_KIND_U = np.array([p[1] == "u" for p in PARAMS])
_A = np.array([p[2] for p in PARAMS])
_B = np.array([p[3] for p in PARAMS])
OBS_NAMES = ["phi2 [deg]", "parallax [mas]", "pm_ra* [mas/yr]", "pm_dec [mas/yr]", "v_los [km/s]"]
OUTLIER_WIDTH = np.array([10.0, 3.0, 4.0, 4.0, 400.0])  # box of the uniform outlier component
N_BINS = 10  # phi1 bins (equal observed counts)
N_MIN_BIN = 20  # model particles a bin needs (StreaMAX-fit's "under-sampled bin" guard)


# ======================================================================================================
# Frames: astropy's default Galactocentric (v4.0) and gala's Pal5PriceWhelan18 stream frame
# ======================================================================================================
def _frame_matrices():
    import astropy.coordinates as ac
    import astropy.units as au
    import gala.coordinates as gco
    from astropy.coordinates.builtin_frames.galactocentric import get_matrix_vectors

    A, off = get_matrix_vectors(ac.Galactocentric(), inverse=True)  # galactocentric -> ICRS
    off_x = off.xyz.to_value(au.kpc)
    off_v = off.differentials["s"].d_xyz.to_value(au.km / au.s)
    eye = ac.SkyCoord(
        ac.CartesianRepresentation(np.eye(3) * au.dimensionless_unscaled), frame="icrs"
    ).transform_to(gco.Pal5PriceWhelan18())
    R = eye.cartesian.xyz.value  # column k = image of the ICRS basis vector e_k
    return np.asarray(A), off_x, off_v, np.asarray(R)


A_GC2ICRS, OFF_X, OFF_V, R_PAL5 = _frame_matrices()


def icrs_to_galcen(ra, dec, dist, pmra, pmdec, vr):
    """Observed ICRS -> Galactocentric (kpc, kpc/Myr)."""
    ra, dec = jnp.deg2rad(ra), jnp.deg2rad(dec)
    cr, sr, cd, sd = jnp.cos(ra), jnp.sin(ra), jnp.cos(dec), jnp.sin(dec)
    rhat = jnp.stack([cd * cr, cd * sr, sd])
    e_ra = jnp.stack([-sr, cr, jnp.zeros_like(ra)])
    e_dec = jnp.stack([-sd * cr, -sd * sr, cd])
    x = dist * rhat
    v = vr * rhat + K_PM * dist * (pmra * e_ra + pmdec * e_dec)  # km/s
    A = jnp.asarray(A_GC2ICRS)
    return A.T @ (x - jnp.asarray(OFF_X)), A.T @ (v - jnp.asarray(OFF_V)) / KMS


def galcen_to_obs(x, v):
    """Galactocentric (N,3) kpc, kpc/Myr -> (N,6) = phi1, phi2 [deg], parallax [mas], pm_ra*, pm_dec [mas/yr], v_los [km/s]."""
    A = jnp.asarray(A_GC2ICRS)
    xi = x @ A.T + jnp.asarray(OFF_X)
    vi = (v * KMS) @ A.T + jnp.asarray(OFF_V)
    d = jnp.linalg.norm(xi, axis=-1)
    rhat = xi / d[:, None]
    ra = jnp.arctan2(xi[:, 1], xi[:, 0])
    dec = jnp.arcsin(rhat[:, 2])
    cr, sr, cd, sd = jnp.cos(ra), jnp.sin(ra), jnp.cos(dec), jnp.sin(dec)
    e_ra = jnp.stack([-sr, cr, jnp.zeros_like(ra)], axis=-1)
    e_dec = jnp.stack([-sd * cr, -sd * sr, cd], axis=-1)
    vlos = jnp.sum(vi * rhat, axis=-1)
    pmra = jnp.sum(vi * e_ra, axis=-1) / (K_PM * d)
    pmdec = jnp.sum(vi * e_dec, axis=-1) / (K_PM * d)
    s = rhat @ jnp.asarray(R_PAL5).T
    phi1 = jnp.rad2deg(jnp.arctan2(s[:, 1], s[:, 0]))
    phi2 = jnp.rad2deg(jnp.arcsin(jnp.clip(s[:, 2], -1.0, 1.0)))
    return jnp.stack([phi1, phi2, 1.0 / d, pmra, pmdec, vlos], axis=-1)


def radec_to_phi(ra_deg, dec_deg):
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    s = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1) @ R_PAL5.T
    return np.rad2deg(np.arctan2(s[:, 1], s[:, 0])), np.rad2deg(np.arcsin(s[:, 2]))


# ======================================================================================================
# Potential and stream generator (galax)
# ======================================================================================================
def host_potential(log10_m_nfw, r_s, q, gl_order: int = GL_ORDER):
    """Cautun+2020 disks + bulge (fixed) + flattened NFW halo (free)."""
    return gp.CompositePotential(
        thin=gp.MN3ExponentialPotential(m_tot=M_THIN, h_R=THIN["Rd"], h_z=THIN["zd"], units=USYS),
        thick=gp.MN3ExponentialPotential(m_tot=M_THICK, h_R=THICK["Rd"], h_z=THICK["zd"], units=USYS),
        bulge=gp.PlummerPotential(m_tot=M_BULGE, r_s=B_BULGE, units=USYS),
        halo=gp.TriaxialNFWPotential(
            m=10.0**log10_m_nfw, r_s=r_s, q1=1.0, q2=q, units=USYS, integration_order=gl_order
        ),
    )


def _solve(acc, x0, v0, t0, t1, rtol, atol, saveat=None, max_steps=16384):
    """Integrate one orbit with diffrax (adaptive Dopri8) -- only used by the accuracy reference."""

    def vf(t, y, _):
        return jnp.concatenate([y[3:], acc(y[:3], t)])

    return dfx.diffeqsolve(
        dfx.ODETerm(vf),
        dfx.Dopri8(),
        t0=t0,
        t1=t1,
        dt0=None,
        y0=jnp.concatenate([x0, v0]),
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        saveat=dfx.SaveAt(t1=True) if saveat is None else dfx.SaveAt(ts=saveat),
        max_steps=max_steps,
        throw=False,
    )


_CHEN_DF = None


def chen_df():
    global _CHEN_DF
    if _CHEN_DF is None:
        _CHEN_DF = gd.ChenStreamDF()
    return _CHEN_DF


def leapfrog(acc, x, v, t0, t1, n_steps: int):
    """Kick-drift-kick leapfrog for a batch of particles, each taking ``n_steps`` equal steps from its own
    ``t0`` to the common ``t1`` (one force evaluation per step; the force may depend on time)."""
    dt = (t1 - t0) / n_steps  # (N,)
    f = jax.vmap(acc)

    def step(i, c):
        x, v, a = c
        v = v + 0.5 * dt[:, None] * a
        x = x + dt[:, None] * v
        a = f(x, t0 + (i + 1) * dt)
        return x, v + 0.5 * dt[:, None] * a, a

    x, v, _ = jax.lax.fori_loop(0, n_steps, step, (x, v, f(x, t0)))
    return x, v


def progenitor_mass(log10_m0, t, t_end):
    """Progenitor mass at time t in [-t_end, 0]: log-linear (as StreaMAX) from the initial mass to today's."""
    return 10.0 ** (log10_m0 + (LOG10_M_NOW - log10_m0) * (1.0 + t / t_end))


def simulate_stream(theta, key=None, n_release: int = N_RELEASE, n_steps: int = N_STEPS, reference: bool = False,
                    progenitor: bool = True, gl_order: int = GL_ORDER):
    """theta (NDIM,) -> (obs (2 n_release, 6), ok, extras); obs columns as in `galcen_to_obs`.

    Default: leapfrog throughout (the StreaMAX scheme) -- the progenitor is initialised by integrating
    today's phase-space point backwards over ``n_release`` steps, and every particle takes ``n_steps`` equal
    KDK steps from its release to today. ``reference=True`` uses adaptive Dopri8 (rtol 1e-9) for both;
    it needs PAL5_X64=1 and is the accuracy reference of `bench`."""
    key = jr.key(STREAM_KEY) if key is None else key
    theta = jnp.asarray(theta, dtype=jnp.result_type(float))
    p = dict(zip(NAMES, theta))
    host = host_potential(p["log10_m_nfw"], p["r_s"], p["q"], gl_order)
    a_prog = p["a_prog"] * 1e-3  # pc -> kpc
    t_end = p["t_end"] * 1e3  # Gyr -> Myr
    acc_host = lambda x, t: -host._gradient(x, t)

    # 1) progenitor: today (t = 0) backwards to -t_end, on n_release + 1 knots
    x_now, v_now = icrs_to_galcen(PAL5_RA, PAL5_DEC, p["dist"], p["pmra"], p["pmdec"], p["vr"])
    tk = -t_end * jnp.linspace(1.0, 0.0, n_release + 1)  # ascending knots, the last one is today
    if reference:
        sol = _solve(acc_host, x_now, v_now, 0.0, -t_end, 1e-10, 1e-10, saveat=tk[::-1])
        xk, vk = sol.ys[::-1, :3], sol.ys[::-1, 3:]
        ok = sol.result == dfx.RESULTS.successful
    else:
        dt = -t_end / n_release

        def back(c, _):
            x, v, a = c
            v = v + 0.5 * dt * a
            x = x + dt * v
            a = acc_host(x, 0.0)
            v = v + 0.5 * dt * a
            return (x, v, a), (x, v)

        _, (xs, vs) = jax.lax.scan(back, (x_now, v_now, acc_host(x_now, 0.0)), None, length=n_release)
        xk = jnp.concatenate([x_now[None], xs])[::-1]
        vk = jnp.concatenate([v_now[None], vs])[::-1]
        ok = jnp.asarray(True)
    ts, xp, vp = tk[:-1], xk[:-1], vk[:-1]  # release times: uniform over [-t_end, 0)
    m_ts = progenitor_mass(p["log10_m_prog"], ts, t_end)

    # 2) Chen+2024 release (galax ChenStreamDF; tidal radius from the HOST potential only).
    #    The DF mixes in G as a Quantity, so it is fed Quantities and its output is stripped again.
    ics = chen_df()._sample(key, host, u.Q(xp, "kpc"), u.Q(vp, "kpc/Myr"), u.Q(m_ts, "Msun"), u.Q(ts, "Myr"))
    x_l, v_l, x_t, v_t = (u.ustrip(unit, a) for a, unit in zip(ics, ("kpc", "kpc/Myr", "kpc", "kpc/Myr")))
    x0 = jnp.stack([x_l, x_t], axis=1).reshape(-1, 3)  # ordered by release time (lead, trail alternate)
    v0 = jnp.stack([v_l, v_t], axis=1).reshape(-1, 3)
    t0 = jnp.repeat(ts, 2)

    # 3) particles in host + the progenitor's own moving, mass-losing Plummer potential. The Plummer
    #    potential is linear in its mass, so a unit-mass one is translated along the orbit and scaled by m(t).
    if progenitor:
        # (= TimeDependentTranslationParameter.from_, minus interpax's input check that cannot be jitted)
        track = gp.params.TimeDependentTranslationParameter(
            translation=interpax.CubicSpline(tk, xk, extrapolate=False, check=False), units=host.units
        )
        unit_prog = gp.TranslatedPotential(gp.PlummerPotential(m_tot=1.0, r_s=a_prog, units=USYS), translation=track)

        def acc(x, t):
            t = jnp.clip(t, -t_end, 0.0)  # round-off must not step outside the spline
            return acc_host(x, t) - progenitor_mass(p["log10_m_prog"], t, t_end) * unit_prog._gradient(x, t)

    else:
        acc = acc_host

    if reference:

        def one(x0i, v0i, t0i):
            s = _solve(acc, x0i, v0i, t0i, 0.0, 1e-9, 1e-9)
            return s.ys[0], s.result == dfx.RESULTS.successful

        y, oks = jax.vmap(one)(x0, v0, t0)
    else:
        xf, vf = leapfrog(acc, x0, v0, t0, 0.0, n_steps)
        y, oks = jnp.concatenate([xf, vf], axis=-1), jnp.asarray(True)
    obs = galcen_to_obs(y[:, :3], y[:, 3:])
    ok = ok & jnp.all(oks) & jnp.all(jnp.isfinite(obs))
    extras = dict(xv=y, prog_x=xk, prog_v=vk, tk=tk, release=t0, m_release=m_ts)
    return obs, ok, extras


# ======================================================================================================
# Data
# ======================================================================================================
def load_data():
    """Observed Pal 5 stream stars (STREAMFINDER members, Gaia DR3 astrometry, DESI/other v_los)."""
    d = np.load(DATA)
    j = int(np.where(d["j"].ravel() == 0)[0][0])  # Pal 5 is stream index 0
    m = d["attention_mask"][j, 0] > 0
    x = d["sim_data_projected"][0, j][m]  # ra, dec, parallax, pmra, pmdec, vlos
    mag = d["magnitudes"][j][m]
    has_v = d["vlos_mask"][j][m] > 0
    v_err = d["vlos_error"][j][m]
    # Gaia DR3 astrometric errors as a function of G (same table as augmentation/streams.py)
    rows = {}
    with open(GAIA_ERR) as f:
        header = f.readline().split("\t")
        grid = np.array([float(h) for h in header[1:-1]])
        for line in f:
            c = line.rstrip("\n").split("\t")
            rows[c[0].strip('"')] = np.array([float(v) for v in c[1:-1]])
    err = lambda k: np.interp(mag, grid, rows[k])
    phi1, phi2 = radec_to_phi(x[:, 0], x[:, 1])
    y = np.stack([phi2, x[:, 2], x[:, 3], x[:, 4], np.where(has_v, x[:, 5], 0.0)], -1)
    e = np.stack(
        [np.hypot(err("ra"), err("dec")) / 3.6e6, err("parallax"), err("mu_ra"), err("mu_dec"), v_err], -1
    )  # position error mas -> deg (negligible)
    mask = np.ones_like(y)
    mask[:, 4] = has_v
    edges = np.quantile(phi1, np.linspace(0, 1, N_BINS + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    b = np.clip(np.searchsorted(edges, phi1) - 1, 0, N_BINS - 1)
    return dict(phi1=phi1, y=y, err=e, mask=mask, edges=edges, bin=b, ra=x[:, 0], dec=x[:, 1], mag=mag)


def bin_data(data, n_min: int = 4):
    """Observed stars -> one value per (phi1 bin, observable), with the SAME statistic as ``bin_model``: a
    linear fit in phi1 inside the bin, centre = value at the mean phi1 + median residual, width = 16-84
    half-width of the residuals (floored by the median measurement error). ``err`` is the standard error
    of that centre, sqrt(pi/2) * width / sqrt(n - 2). Cells with fewer than ``n_min`` stars are masked.
    The result has the layout of ``load_data`` with one row per bin (phi1 is per cell: v_los uses fewer stars)."""
    B = N_BINS
    y, e, m, ph, cnt, wid = (np.zeros((B, 5)) for _ in range(6))
    e += 1.0
    for bb in range(B):
        for d in range(5):
            s = (data["bin"] == bb) & (data["mask"][:, d] > 0)
            n = int(s.sum())
            cnt[bb, d] = n
            ph[bb, d] = data["phi1"][data["bin"] == bb].mean()
            if n < n_min:
                continue
            p_, v = data["phi1"][s], data["y"][s, d]
            dp = p_ - p_.mean()
            slope = (dp * (v - v.mean())).sum() / max((dp**2).sum(), 1e-12)
            q = np.quantile(v - v.mean() - slope * dp, [0.16, 0.5, 0.84])
            sig = max(0.5 * (q[2] - q[0]), float(np.median(data["err"][s, d])))
            y[bb, d], wid[bb, d], ph[bb, d], m[bb, d] = v.mean() + q[1], sig, p_.mean(), 1.0
            e[bb, d] = np.sqrt(0.5 * np.pi / (n - 2)) * sig
    return dict(phi1=ph, y=y, err=e, mask=m, edges=data["edges"], bin=np.arange(B), n_obs=cnt, width_obs=wid, stars=data)


# ======================================================================================================
# Likelihood and prior
# ======================================================================================================
def bin_model(obs, edges):
    """Per phi1 bin: particle count, mean phi1 and, for each observable, the local linear track (value at
    the mean phi1 + slope) with a robust centre and width -- the median and the 16-84 half-width of the
    residuals about that line, as StreaMAX's ``get_track_2D`` does for its radial track."""
    phi1, y = obs[:, 0], obs[:, 1:]
    nb = edges.shape[0] - 1
    idx = jnp.searchsorted(edges, phi1) - 1
    inbin = (idx[:, None] == jnp.arange(nb)[None, :]) & (phi1[:, None] > edges[0]) & (phi1[:, None] < edges[-1])
    w = inbin.astype(obs.dtype)  # (N, B)
    n = w.sum(0)
    nn = jnp.maximum(n, 1.0)
    mphi = (w * phi1[:, None]).sum(0) / nn
    dphi = phi1[:, None] - mphi[None, :]  # (N, B)
    vphi = (w * dphi**2).sum(0) / nn
    my = jnp.einsum("nb,nd->bd", w, y) / nn[:, None]
    dy = y[:, None, :] - my[None, :, :]  # (N, B, D)
    cov = jnp.einsum("nb,nb,nbd->bd", w, dphi, dy) / nn[:, None]
    slope = cov / jnp.maximum(vphi, 1e-12)[:, None]
    res = jnp.where(inbin[:, :, None], dy - slope[None] * dphi[:, :, None], jnp.nan)
    q = jnp.nanquantile(res, jnp.asarray([0.16, 0.5, 0.84], dtype=obs.dtype), axis=0)  # (3, B, D)
    centre = jnp.nan_to_num(my + q[1])
    sig = jnp.nan_to_num(0.5 * (q[2] - q[0]))
    return n, mphi, centre, slope, sig**2


def make_loglike(data, n_release: int = N_RELEASE, n_steps: int = N_STEPS, binned: bool = False):
    """``binned=False``: every observed star against the model's bin track (variance = model width^2 + Gaia
    error^2 + sigma_inflate^2, outlier mixture). ``binned=True`` (``data`` from ``bin_data``): bin against
    bin -- the observed bin centre against the model bin centre evaluated at the same phi1, variance =
    standard error of the observed centre^2 + Monte-Carlo error of the model centre^2 + sigma_inflate^2.
    There sigma_inflate is a per-bin TRACK systematic (it does not average down with the number of stars),
    widths are not compared and f_out is unused (its posterior is its prior)."""
    dtype = jnp.result_type(float)
    phi1 = jnp.asarray(data["phi1"], dtype)
    y = jnp.asarray(data["y"], dtype)
    err2 = jnp.asarray(data["err"], dtype) ** 2
    mask = jnp.asarray(data["mask"], dtype)
    edges = jnp.asarray(data["edges"], dtype)
    b = jnp.asarray(data["bin"])
    log_out = -jnp.sum(mask * jnp.log(jnp.asarray(OUTLIER_WIDTH, dtype)), axis=-1)

    def predict(theta, obs):
        """Per-star predictive mean and total variance (N_star, 5), the model-only variance, bin counts."""
        p = dict(zip(NAMES, theta))
        sig2 = jnp.stack([p["sig_phi2"], p["sig_plx"], p["sig_pmra"], p["sig_pmdec"], p["sig_vlos"]]) ** 2
        n, mphi, centre, slope, var = bin_model(obs, edges)
        if binned:
            mu = centre[b] + slope[b] * (phi1 - mphi[b][:, None])
            tot = var[b] * (0.5 * jnp.pi / jnp.maximum(n[b], 1.0))[:, None] + err2 + sig2[None, :]
            return mu, tot, var[b], n
        mu = centre[b] + slope[b] * (phi1 - mphi[b])[:, None]
        # model dispersion (+ the Monte-Carlo error of its median) + Gaia error + sampled inflation
        tot = var[b] * (1.0 + 0.5 * jnp.pi / jnp.maximum(n[b], 1.0))[:, None] + err2 + sig2[None, :]
        return mu, tot, var[b], n

    def loglike_of(theta, mu, tot, n, yy):
        logn = jnp.sum(mask * (-0.5 * (yy - mu) ** 2 / tot - 0.5 * jnp.log(2 * jnp.pi * tot)), axis=-1)
        if binned:
            return jnp.sum(logn)
        f = theta[NAMES.index("f_out")]
        return jnp.sum(jnp.logaddexp(jnp.log1p(-f) + logn, jnp.log(f) + log_out))

    def loglike_from_obs(theta, obs, ok):
        mu, tot, _, n = predict(theta, obs)
        ll = loglike_of(theta, mu, tot, n, y)
        # StreaMAX-fit guard: a model that leaves an observed bin under-sampled is not compared with the
        # data at all; the penalty grows with the shortfall so the sampler is pushed back, not walled off.
        short = jnp.sum(jnp.maximum(N_MIN_BIN - n, 0.0)) / N_MIN_BIN
        ll = jnp.where(short > 0, -1e6 * (1.0 + short), ll)
        return jnp.where(ok & jnp.isfinite(ll), ll, -1e30).astype(dtype)

    def loglike(theta):
        obs, ok, _ = simulate_stream(theta, n_release=n_release, n_steps=n_steps)
        return loglike_from_obs(jnp.asarray(theta, dtype), obs, ok)

    loglike.predict, loglike.of = predict, loglike_of
    return loglike, loglike_from_obs


BSPLINE_OPS = OUT / "sbi_comparison/nsbspline_1000ep/bspline_ops.npz"  # H (5, 20, n_star), float64


def bspline_ops_file(out_dir: Path) -> Path:
    """The spline operators of a run: ``<out_dir>/bspline_ops.npz``, copied once from ``BSPLINE_OPS``."""
    dst = Path(out_dir) / "bspline_ops.npz"
    if not dst.exists():
        import shutil

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".bspline_ops.{os.getpid()}.tmp")
        shutil.copyfile(BSPLINE_OPS, tmp)
        os.replace(tmp, dst)  # atomic: workers starting together never read a partial file
    return dst


def bspline_row_space(ops_path, rtol: float = 1e-10):
    """Orthonormal bases V_k (n_star, r_k) of the row spaces of the spline operators H[k] (rank 15, 15, 15,
    15, 10): z_k = V_k^T y_k is a one-to-one, well-conditioned re-expression of the track H[k] @ y_k."""
    H = np.asarray(np.load(ops_path)["H"], np.float64)
    V = []
    for k in range(H.shape[0]):
        _, sv, vt = np.linalg.svd(H[k], full_matrices=False)
        V.append(vt[sv > rtol * sv[0]].T)
    return V


def make_loglike_bspline(data, ops_path=BSPLINE_OPS, n_release: int = N_RELEASE, n_steps: int = N_STEPS):
    """B-spline-track likelihood (``data`` from ``load_data``). The stars are Gaussian about the bin track
    with the star likelihood's mean and variance (``make_loglike(binned=False)``, NO outlier term); the data
    are their penalised-spline tracks H[k] @ y[:, k]. H[k] is rank deficient, so the density is evaluated in
    the row space:  logL = sum_k log N(V_k^T y_k; V_k^T mu_k, V_k^T diag(tot_k) V_k)  (70 dimensions).
    f_out is unused (its posterior is its prior). Same veto as ``make_loglike``. The same density as
    ``pal5_sbi_comparison.NoiseModel.logpdf`` with ``tracks=True``."""
    assert X64, "the B-spline-track likelihood needs float64"
    star_loglike, _ = make_loglike(data, n_release=n_release, n_steps=n_steps, binned=False)
    predict = star_loglike.predict
    dtype = jnp.result_type(float)
    V_np = bspline_row_space(ops_path)
    assert all(v.shape[0] == len(data["phi1"]) for v in V_np), "spline operators and stars do not match"
    V = [jnp.asarray(v, dtype) for v in V_np]
    y = jnp.asarray(data["y"], dtype)

    def loglike_of(theta, mu, tot, n, yy):
        out = jnp.zeros((), dtype)
        for k, v in enumerate(V):
            r = (yy[:, k] - mu[:, k]) @ v  # (r_k,)
            chol = jnp.linalg.cholesky((v * tot[:, k][:, None]).T @ v)
            a = jax.scipy.linalg.solve_triangular(chol, r, lower=True)
            out += -0.5 * jnp.sum(a * a) - jnp.sum(jnp.log(jnp.diagonal(chol))) - 0.5 * v.shape[1] * jnp.log(2 * jnp.pi)
        return out

    def loglike_from_obs(theta, obs, ok):
        mu, tot, _, n = predict(theta, obs)
        ll = loglike_of(theta, mu, tot, n, y)
        short = jnp.sum(jnp.maximum(N_MIN_BIN - n, 0.0)) / N_MIN_BIN  # the veto of `make_loglike`
        ll = jnp.where(short > 0, -1e6 * (1.0 + short), ll)
        return jnp.where(ok & jnp.isfinite(ll), ll, -1e30).astype(dtype)

    def loglike(theta):
        obs, ok, _ = simulate_stream(theta, n_release=n_release, n_steps=n_steps)
        return loglike_from_obs(jnp.asarray(theta, dtype), obs, ok)

    loglike.predict, loglike.of = predict, loglike_of
    return loglike, loglike_from_obs


# Rotation curves that can be fitted together with the stream (``--rotation-curve``): name -> csv (R, V_c, e_V_c
# or R, V_c, sigma_minus, sigma_plus)
ROT_CURVES = {"jiao": REPO / "assets/rotation_curve_custom_v4.csv",  # 19 radii 6.6-25.5 kpc (the repo's "Jiao26" table)
              "eilers19": REPO / "assets/rotation_curve_eilers19.csv"}  # 38 radii 5.3-24.8 kpc, asymmetric errors


def load_rotation_curve(name: str):
    """R [kpc], v_c [km/s], sigma [km/s] of a table in ``ROT_CURVES``."""
    t = np.array([ln.split(",") for ln in ROT_CURVES[name].read_text().splitlines() if ln[:1].isdigit()], float)
    return t[:, 0], t[:, 1], t[:, 2:].mean(1)  # asymmetric errors (sigma_minus, sigma_plus) are averaged


def model_vcirc(theta, radii):
    """Circular velocity [km/s] of the full host potential (fixed disks + bulge, sampled halo) in the midplane."""
    host = host_potential(theta[0], theta[1], theta[2])
    return jax.vmap(lambda R: jnp.sqrt(R * host._gradient(jnp.stack([R, 0.0 * R, 0.0 * R]), 0.0)[0]) * KMS)(radii)


def make_loglike_rc(name: str):
    """Gaussian log-likelihood of an observed rotation curve (independent points, quoted errors) given theta."""
    dtype = jnp.result_type(float)
    radii, vc, sig = (jnp.asarray(a, dtype) for a in load_rotation_curve(name))

    def loglike(theta):
        r = (model_vcirc(jnp.asarray(theta, dtype), radii) - vc) / sig
        return jnp.sum(-0.5 * r * r - jnp.log(sig) - 0.5 * jnp.log(2 * jnp.pi))

    return loglike


def add_rotation_curve(loglike, name: str):
    """Stream log-likelihood + rotation-curve log-likelihood (the stream veto values stay hugely negative)."""
    rc = make_loglike_rc(name)

    def total(theta):
        ll = rc(theta)
        return loglike(theta) + jnp.where(jnp.isfinite(ll), ll, -1e30)

    return total


def logprior(theta):
    dtype = jnp.result_type(float)
    a, b, ku = jnp.asarray(_A, dtype), jnp.asarray(_B, dtype), jnp.asarray(_KIND_U)
    lu = jnp.where((theta >= a) & (theta <= b), -jnp.log(b - a), -jnp.inf)
    ln = -0.5 * ((theta - a) / b) ** 2 - jnp.log(b * jnp.sqrt(2 * jnp.pi))
    return jnp.sum(jnp.where(ku, lu, ln))


def prior_sample(key, n):
    dtype = jnp.result_type(float)
    ku, kn = jr.split(key)
    uu = jr.uniform(ku, (n, NDIM), dtype)
    z = jr.normal(kn, (n, NDIM), dtype)
    a, b = jnp.asarray(_A, dtype), jnp.asarray(_B, dtype)
    return jnp.where(jnp.asarray(_KIND_U), a + (b - a) * uu, a + b * z)


FIDUCIAL = np.array([11.75, 16.0, 1.0, 20.6, -2.736, -2.646, -58.6, 4.5, 21.0, 4.0, 0.25, 0.2, 0.1, 0.1, 5.0, 0.03])


# ======================================================================================================
# Nested sampling (blackjax nested slice sampling, batch-synchronous slice update)
# ======================================================================================================
def sync_slice_update(loglike_batch, logprior_fn, max_steps: int = 10, max_shrinkage: int = 100):
    """``update_strategy`` for ``blackjax.nss``: the same constrained hit-and-run slice move as blackjax's
    default (direction ~ live covariance with Mahalanobis norm 2, unit bracket, stepping out, shrinkage,
    validity = inside the prior slice AND logL > logL*), but all ``num_delete`` chains advance in lock-step,
    ONE batched likelihood call per loop iteration. blackjax's default vmaps whole chains, so every slice
    costs the batch as many likelihood calls as its slowest member needs (~2x more for this problem).
    Stepping out follows Neal (2003, fig. 3): at most ``max_steps`` expansions, split at random between
    the two sides."""
    import blackjax.ns.base as nsb

    def strategy(constrained_step_fn, num_inner_steps, num_delete):
        del constrained_step_fn

        def update(rng_key, state, loglikelihood_0, covariance_factor):
            parts = state.particles
            d = covariance_factor.shape[-1]
            k_choice, k_init, k_loop = jr.split(rng_key, 3)
            alive = (parts.loglikelihood > loglikelihood_0).astype(float)
            alive = jnp.where(alive.sum() > 0, alive, jnp.ones_like(alive))
            idx = jr.choice(k_choice, alive.shape[0], (num_delete,), p=alive / alive.sum(), replace=True)
            x, lp, ll = parts.position[idx], parts.logdensity[idx], parts.loglikelihood[idx]

            def new_slice(key, lp_i):
                kd, ky, kb, kj = jr.split(key, 4)
                z = jr.normal(kd, (d,))
                dirn = 2.0 * (covariance_factor @ z) / jnp.linalg.norm(z)
                logy = lp_i + jnp.log(jr.uniform(ky))  # slice level under the prior density
                left = -jr.uniform(kb)
                j = jnp.floor(max_steps * jr.uniform(kj))  # expansions allowed to the left
                return dirn, logy, left, left + 1.0, j, max_steps - 1.0 - j

            def fresh(key, lp):
                return jax.vmap(new_slice)(jr.split(key, num_delete), lp)

            dirn, logy, left, right, jl, jr_ = fresh(k_init, lp)
            zi = jnp.zeros(num_delete, dtype=jnp.int32)
            carry = (x, lp, ll, dirn, logy, left, right, jl, jr_, zi, zi, zi, zi, zi, k_loop, jnp.int32(0))

            def cond(c):
                return jnp.any(c[11] < num_inner_steps) & (c[15] < num_inner_steps * (max_steps + 40))

            def body(c):
                x, lp, ll, dirn, logy, left, right, jl, jr_, phase, nshr, done, nev, nacc, key, it = c
                key, ku, kn = jr.split(key, 3)
                active = done < num_inner_steps
                t = jnp.where(phase == 0, left, jnp.where(phase == 1, right, left + jr.uniform(ku, (num_delete,)) * (right - left)))
                xq = x + t[:, None] * dirn
                lpq = jax.vmap(logprior_fn)(xq)
                llq = loglike_batch(xq)
                valid = (lpq > logy) & (llq > loglikelihood_0)
                # stepping out, left then right
                grow_l = (phase == 0) & valid & (jl > 0)
                grow_r = (phase == 1) & valid & (jr_ > 0)
                left = jnp.where(grow_l, left - 1.0, left)
                jl = jnp.where(grow_l, jl - 1.0, jl)
                right = jnp.where(grow_r, right + 1.0, right)
                jr_ = jnp.where(grow_r, jr_ - 1.0, jr_)
                # shrinkage
                shr = phase == 2
                accept = shr & valid
                give_up = shr & ~valid & (nshr + 1 >= max_shrinkage)  # chain stays put, step is spent
                shrink = shr & ~valid & ~give_up
                left = jnp.where(shrink & (t < 0), t, left)
                right = jnp.where(shrink & (t >= 0), t, right)
                nshr = jnp.where(shrink, nshr + 1, nshr)
                new_phase = jnp.where((phase == 0) & ~grow_l, 1, jnp.where((phase == 1) & ~grow_r, 2, phase))
                # a finished slice (accepted or abandoned) starts the next one
                x = jnp.where((accept & active)[:, None], xq, x)
                lp = jnp.where(accept & active, lpq, lp)
                ll = jnp.where(accept & active, llq, ll)
                fin = accept | give_up
                n_dirn, n_logy, n_left, n_right, n_jl, n_jr = fresh(kn, lp)
                sel = lambda new, old: jnp.where(fin.reshape((-1,) + (1,) * (old.ndim - 1)), new, old)
                dirn, logy, left, right = sel(n_dirn, dirn), sel(n_logy, logy), sel(n_left, left), sel(n_right, right)
                jl, jr_ = sel(n_jl, jl), sel(n_jr, jr_)
                phase = jnp.where(fin, 0, new_phase).astype(jnp.int32)
                nshr = jnp.where(fin, 0, nshr)
                nev = nev + active.astype(jnp.int32)
                nacc = nacc + (accept & active).astype(jnp.int32)
                done = done + (fin & active).astype(jnp.int32)
                return (x, lp, ll, dirn, logy, left, right, jl, jr_, phase, nshr, done, nev, nacc, key, it + 1)

            c = jax.lax.while_loop(cond, body, carry)
            new = nsb.StateWithLogLikelihood(c[0], c[1], c[2], jnp.full_like(c[2], loglikelihood_0))
            return new, dict(n_evals=c[12], n_accept=c[13], n_done=c[11], n_batches=jnp.full((num_delete,), c[15]))

        return update

    return strategy


def live_cov_factor(rng_key, state, info, params=None):
    """Cholesky factor of the live-point covariance for the slice directions (blackjax's
    ``live_covariance_factor`` plus a relative jitter and a diagonal fallback: an (almost) singular
    covariance otherwise gives a NaN factor, NaN directions, and slices that never finish)."""
    del rng_key, info, params
    cov = jnp.atleast_2d(jnp.cov(state.particles.position, rowvar=False))
    dg = jnp.diag(cov)
    fac = jnp.linalg.cholesky(cov + 1e-6 * jnp.diag(dg))
    return {"covariance_factor": jnp.where(jnp.all(jnp.isfinite(fac)), fac, jnp.diag(jnp.sqrt(dg)))}


def toy_problem():
    """Gaussian likelihood on the real prior (analytic evidence) -- the sampler self-test."""
    from scipy.stats import norm

    mu = np.where(_KIND_U, _A + 0.4 * (_B - _A), _A + 0.5 * _B)
    sg = np.where(_KIND_U, 0.05 * (_B - _A), 0.5 * _B)
    logz = 0.0
    for k in range(NDIM):
        if _KIND_U[k]:  # int U(a,b) N(x|mu,sg) dx
            logz += np.log((norm.cdf(_B[k], mu[k], sg[k]) - norm.cdf(_A[k], mu[k], sg[k])) / (_B[k] - _A[k]))
        else:  # int N(x|a,b) N(x|mu,sg) dx
            logz += norm.logpdf(mu[k], _A[k], np.hypot(_B[k], sg[k]))
    mu_j, sg_j = jnp.asarray(mu), jnp.asarray(sg)
    ll = lambda th: jnp.sum(-0.5 * ((th - mu_j) / sg_j) ** 2 - jnp.log(sg_j * jnp.sqrt(2 * jnp.pi)))
    post_mu = np.where(_KIND_U, mu, (mu / sg**2 + _A / _B**2) / (1 / sg**2 + 1 / _B**2))
    post_sg = np.where(_KIND_U, sg, 1 / np.sqrt(1 / sg**2 + 1 / _B**2))
    return ll, float(logz), post_mu, post_sg


def run_ns(loglike, key, nlive, num_delete, num_inner, ckpt: Path | None = None, log=print, dlogz=-5.0,
           max_hours: float = 1e9):
    """One nested-sampling run. Returns dict(theta, logl, logl_birth) of dead + final live points."""
    import blackjax

    algo = blackjax.nss(
        logprior_fn=logprior,
        loglikelihood_fn=loglike,
        num_inner_steps=num_inner,
        num_delete=num_delete,
        update_strategy=sync_slice_update(jax.vmap(loglike), logprior),
        inner_kernel_params=live_cov_factor,
    )
    step = jax.jit(algo.step)
    dead = dict(theta=[], logl=[], logl_birth=[])
    n_evals = 0
    t_start = time.time()
    if ckpt is not None and ckpt.exists():
        import blackjax.ns.adaptive as nsa
        import blackjax.ns.base as nsb
        import blackjax.ns.integrator as nsi
        z = np.load(ckpt)
        parts = nsb.StateWithLogLikelihood(*(jnp.asarray(z[f"live_{k}"]) for k in ("theta", "logp", "logl", "logl_birth")))
        base = nsb.NSState(parts)
        state = nsa.AdaptiveNSState(
            particles=parts,
            inner_kernel_params=live_cov_factor(None, base, None, {}),
            integrator=nsi.NSIntegrator(*(jnp.asarray(z[f"int_{k}"]) for k in ("logX", "logZ", "logZ_live"))),
        )
        for k in dead:
            dead[k] = [z[f"dead_{k}"]]
        key = jr.wrap_key_data(jnp.asarray(z["key"]))
        n_evals = int(z["n_evals"])
        log(f"resumed from {ckpt}: {len(dead['logl'][0])} dead points, logZ = {float(state.integrator.logZ):.3f}")
    else:
        key, k0, k1 = jr.split(key, 3)
        t0 = time.time()
        state = algo.init(prior_sample(k0, nlive), rng_key=k1)
        n_evals = nlive
        ll0 = np.asarray(state.particles.loglikelihood)
        log(f"initialised {nlive} live points in {time.time() - t0:.0f}s: logL min/median/max = "
            f"{ll0.min():.4g}/{np.median(ll0):.4g}/{ll0.max():.4g}; simulation failures {int((ll0 < -1e29).sum())}, "
            f"under-sampled {int(((ll0 < -1e5) & (ll0 > -1e29)).sum())}")
    it = 0
    while not float(state.integrator.logZ_live - state.integrator.logZ) < dlogz:
        if (time.time() - t_start) / 3600 > max_hours:
            log("max_hours reached: stopping early (the remaining live points are appended; evidence truncated)")
            break
        key, sub = jr.split(key)
        t0 = time.time()
        state, info = step(sub, state)
        ui = info.update_info
        dead["theta"].append(np.asarray(info.particles.position))
        dead["logl"].append(np.asarray(info.particles.loglikelihood))
        dead["logl_birth"].append(np.asarray(info.particles.loglikelihood_birth))
        nb = int(ui["n_batches"][0])
        n_evals += nb * num_delete
        it += 1
        log(f"it {it:4d} | dead {sum(len(a) for a in dead['logl']):6d} | logL* {float(info.particles.loglikelihood.max()):11.3f} "
            f"| logZ {float(state.integrator.logZ):10.3f} | logZ_live-logZ {float(state.integrator.logZ_live - state.integrator.logZ):8.2f} "
            f"| {nb} batches, {float(np.mean(ui['n_evals'])) / num_inner:.1f} evals/slice, "
            f"unfinished {int((np.asarray(ui['n_done']) < num_inner).sum())} | {time.time() - t0:.0f}s | evals {n_evals}")
        if ckpt is not None:
            p = state.particles
            tmp = ckpt.with_suffix(".tmp.npz")
            np.savez(
                tmp,
                live_theta=np.asarray(p.position), live_logp=np.asarray(p.logdensity), live_logl=np.asarray(p.loglikelihood),
                live_logl_birth=np.asarray(p.loglikelihood_birth),
                int_logX=np.asarray(state.integrator.logX), int_logZ=np.asarray(state.integrator.logZ),
                int_logZ_live=np.asarray(state.integrator.logZ_live),
                **{f"dead_{k}": np.concatenate(v) for k, v in dead.items()},
                key=np.asarray(jr.key_data(key)), n_evals=n_evals,
            )
            os.replace(tmp, ckpt)
    p = state.particles
    out = dict(
        theta=np.concatenate(dead["theta"] + [np.asarray(p.position)]),
        logl=np.concatenate(dead["logl"] + [np.asarray(p.loglikelihood)]),
        logl_birth=np.concatenate(dead["logl_birth"] + [np.asarray(p.loglikelihood_birth)]),
        n_evals=n_evals, nlive=nlive, hours=(time.time() - t_start) / 3600,
    )
    return out


def ns_weights(logl, logl_birth, key, n_sim: int = 200):
    """Posterior log-weights and evidence of a (possibly merged) run from its birth/death contours.
    Returns (logw mean (n,), logZ samples (n_sim,))."""
    import blackjax.ns.base as nsb
    import blackjax.ns.utils as nsu
    from jax.scipy.special import logsumexp

    parts = nsb.StateWithLogLikelihood(jnp.zeros_like(jnp.asarray(logl)), jnp.zeros_like(jnp.asarray(logl)),
                                       jnp.asarray(logl), jnp.asarray(logl_birth))
    lw = nsu.log_weights(key, nsb.NSInfo(parts, None), shape=n_sim)  # (n, n_sim)
    logz = logsumexp(lw, axis=0)
    return np.asarray(logsumexp(lw - logz[None, :], axis=1) - np.log(n_sim)), np.asarray(logz)


def cmd_selftest(args):
    """Sampler check on a Gaussian toy with the real prior: evidence and posterior moments vs analytic."""
    ll, logz_true, mu, sg = toy_problem()
    out = run_ns(ll, jr.key(args.seed), args.nlive, args.num_delete, args.num_inner or 2 * NDIM, log=lambda m: None)
    lw, logz = ns_weights(out["logl"], out["logl_birth"], jr.key(1))
    w = np.exp(lw - lw.max())
    w /= w.sum()
    m = (w[:, None] * out["theta"]).sum(0)
    sd = np.sqrt((w[:, None] * (out["theta"] - m) ** 2).sum(0))
    print(f"selftest: logZ = {logz.mean():.3f} +- {logz.std():.3f}  (analytic {logz_true:.3f});  "
          f"ESS {1 / np.sum(w**2):.0f}; evals {out['n_evals']}")
    print("  max |mean - true| / sigma_true:", np.abs((m - mu) / sg).max().round(3),
          "  max |sd/sd_true - 1|:", np.abs(sd / sg - 1).max().round(3))


def cmd_worker(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"run{args.seed:02d}"
    logf = open(out_dir / f"{tag}.log", "a")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"worker seed {args.seed} on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} ({jax.devices()}); x64={X64}; "
        f"nlive {args.nlive}, num_delete {args.num_delete}, num_inner {args.num_inner or 2 * NDIM}; "
        f"n_release {args.n_release} (x2 particles), n_steps {args.n_steps}")
    binned = args.likelihood == "bin"
    log(f"likelihood: {args.likelihood}")
    data = bin_data(load_data()) if binned else load_data()
    if args.likelihood == "bspline":
        loglike, _ = make_loglike_bspline(data, bspline_ops_file(out_dir), n_release=args.n_release, n_steps=args.n_steps)
    else:
        loglike, _ = make_loglike(data, n_release=args.n_release, n_steps=args.n_steps, binned=binned)
    if args.rotation_curve:
        log(f"rotation curve fitted jointly: {args.rotation_curve} ({ROT_CURVES[args.rotation_curve].name})")
        loglike = add_rotation_curve(loglike, args.rotation_curve)
    out = run_ns(loglike, jr.key(1000 + args.seed), args.nlive, args.num_delete, args.num_inner or 2 * NDIM,
                 ckpt=out_dir / f"{tag}_ckpt.npz", log=log, max_hours=args.max_hours)
    np.savez(out_dir / f"{tag}_final.npz", **out)
    log(f"finished: {len(out['logl'])} points, {out['n_evals']} likelihood evaluations, {out['hours']:.2f} h (this session)")


def cmd_run(args):
    """Launch one independent run per free GPU (autocvd picks them, waiting if needed) and wait."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu_ids:  # chosen by hand (indices or UUIDs), e.g. where autocvd crashes on a broken card
        gpus = args.gpu_ids.split(",")
        print(f"GPUs given with --gpu-ids: {gpus}", flush=True)
    else:
        gpus = pick_free_gpus(args.gpus)
        print(f"free GPUs from autocvd: {gpus}", flush=True)
    if args.likelihood == "bspline":
        bspline_ops_file(out_dir)
    procs = []
    for seed, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        env.pop("JAX_PLATFORMS", None)
        cmd = [sys.executable, __file__, "worker", "--seed", str(seed), "--out", str(out_dir), "--nlive", str(args.nlive),
               "--num-delete", str(args.num_delete), "--num-inner", str(args.num_inner), "--n-release", str(args.n_release),
               "--n-steps", str(args.n_steps), "--max-hours", str(args.max_hours), "--likelihood", args.likelihood,
               "--rotation-curve", args.rotation_curve]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(out_dir / f"run{seed:02d}.stdout", "a"), stderr=subprocess.STDOUT))
        print(f"  run{seed:02d} -> GPU {gpu} (pid {procs[-1].pid})", flush=True)
    codes = [p.wait() for p in procs]
    print("exit codes:", codes, flush=True)
    sys.exit(max(codes))


def _timeit(f, *a):
    r = f(*a)
    jax.block_until_ready(r)
    t0 = time.time()
    r = f(*a)
    jax.block_until_ready(r)
    return r, time.time() - t0


def _compare_bins(a, b, edges):
    """max over bins of |centre shift| / sigma_b and of |sigma ratio - 1| between two particle sets."""
    na, _, ca, _, va = (np.asarray(v) for v in bin_model(jnp.asarray(a), edges))
    nb, _, cb, _, vb = (np.asarray(v) for v in bin_model(jnp.asarray(b), edges))
    good = (na >= N_MIN_BIN) & (nb >= N_MIN_BIN)
    return (np.abs(ca - cb) / np.sqrt(vb))[good].max(0), np.abs(np.sqrt(va / vb) - 1)[good].max(0), int(good.sum())


def cmd_bench(args):
    """Forward-model checks: accuracy vs an adaptive Dopri8 reference, a gala cross-check of the generator,
    the likelihood scatter over spray realisations, and the throughput."""
    data = load_data()
    loglike, lfo = make_loglike(data, n_release=args.n_release, n_steps=args.n_steps)
    edges = jnp.asarray(data["edges"])
    th = jnp.asarray(FIDUCIAL)
    print(f"device {jax.devices()}, x64 {X64}; M_thin {M_THIN:.4e}, M_thick {M_THICK:.4e}, M_bulge {M_BULGE:.4e} Msun "
          f"(Plummer b = {B_BULGE:.3f} kpc); {len(data['phi1'])} stars, {int(data['mask'][:, 4].sum())} with v_los")

    # 1) leapfrog vs adaptive reference
    sim = jax.jit(lambda t: simulate_stream(t, n_release=args.n_release, n_steps=args.n_steps)[:2])
    (obs, ok), dt = _timeit(sim, th)
    ref, okr = jax.jit(lambda t: simulate_stream(t, n_release=args.n_release, reference=True)[:2])(th)
    sh, sr, nb = _compare_bins(obs, ref, edges)
    print(f"[accuracy] leapfrog ({args.n_steps} steps, {dt:.2f}s) vs adaptive Dopri8 rtol 1e-9, {nb} bins:")
    print(f"   max |bin-centre shift| / sigma_bin per observable: {sh.round(3)}")
    print(f"   max |sigma ratio - 1|              per observable: {sr.round(3)}")
    print(f"   logL leapfrog {float(lfo(th, obs, ok)):.3f}  vs  reference {float(lfo(th, ref, okr)):.3f}")

    # 2) gala cross-check (same potential, spherical halo, constant progenitor mass, independent RNG)
    import astropy.units as au
    import gala.dynamics as gdy
    import gala.dynamics.mockstream as gms
    import gala.potential as gpo
    from gala.units import galactic

    thg = np.array(FIDUCIAL)
    thg[NAMES.index("log10_m_prog")] = LOG10_M_NOW
    p = dict(zip(NAMES, thg))
    obs_g, _, ex = simulate_stream(jnp.asarray(thg), n_release=args.n_release, n_steps=args.n_steps)
    pot = gpo.CCompositePotential()
    pot["thin"] = gpo.MN3ExponentialDiskPotential(m=M_THIN, h_R=THIN["Rd"], h_z=THIN["zd"], positive_density=False, sech2_z=False, units=galactic)
    pot["thick"] = gpo.MN3ExponentialDiskPotential(m=M_THICK, h_R=THICK["Rd"], h_z=THICK["zd"], positive_density=False, sech2_z=False, units=galactic)
    pot["bulge"] = gpo.PlummerPotential(m=M_BULGE, b=B_BULGE, units=galactic)
    pot["halo"] = gpo.NFWPotential(m=10 ** p["log10_m_nfw"], r_s=p["r_s"], units=galactic)
    x_now, v_now = (np.asarray(a) for a in icrs_to_galcen(PAL5_RA, PAL5_DEC, p["dist"], p["pmra"], p["pmdec"], p["vr"]))
    host = host_potential(p["log10_m_nfw"], p["r_s"], 1.0)
    a_galax = -np.asarray(host._gradient(jnp.asarray(x_now), 0.0))
    a_gala = pot.acceleration(x_now * au.kpc).to_value(au.kpc / au.Myr**2)[:, 0]
    print(f"[gala] host acceleration at Pal 5: galax {a_galax}, gala {a_gala}, max rel diff {np.abs(a_galax / a_gala - 1).max():.2e}")
    w0 = gdy.PhaseSpacePosition(pos=x_now * au.kpc, vel=v_now * au.kpc / au.Myr)
    gen = gms.MockStreamGenerator(gms.ChenStreamDF(), gpo.Hamiltonian(pot),
                                  progenitor_potential=gpo.PlummerPotential(m=10**LOG10_M_NOW, b=p["a_prog"] * 1e-3, units=galactic))
    n_g = 4000
    stream, _ = gen.run(w0, 10**LOG10_M_NOW * au.Msun, dt=-(p["t_end"] * 1e3 / n_g) * au.Myr, n_steps=n_g)
    obs_gala = galcen_to_obs(jnp.asarray(stream.xyz.to_value(au.kpc).T), jnp.asarray(stream.v_xyz.to_value(au.kpc / au.Myr).T))
    sh, sr, nb = _compare_bins(obs_g, obs_gala, edges)
    print(f"[gala] galax-assembled Chen+24 stream vs gala MockStreamGenerator(ChenStreamDF, progenitor_potential), "
          f"{obs_gala.shape[0]} gala particles, {nb} bins (two independent random realisations):")
    print(f"   max |bin-centre shift| / sigma_bin: {sh.round(3)}")
    print(f"   max |sigma ratio - 1|             : {sr.round(3)}")

    # 3) likelihood scatter over the spray realisation (the sampler uses ONE fixed key)
    keys = jr.split(jr.key(7), 24)
    ll_k = jax.jit(jax.vmap(lambda k: lfo(th, *simulate_stream(th, key=k, n_release=args.n_release, n_steps=args.n_steps)[:2])))(keys)
    print(f"[noise] logL over 24 spray keys at the fiducial point: mean {float(ll_k.mean()):.2f}, std {float(ll_k.std()):.2f}")

    # 4) throughput
    bl = jax.jit(jax.vmap(loglike))
    thetas = prior_sample(jr.key(1), 64)
    for nbatch in (1, 16, 40, 64):
        r, dt = _timeit(bl, thetas[:nbatch])
        r = np.asarray(r)
        print(f"[speed] batch {nbatch:3d}: {dt:6.2f}s = {dt / nbatch:.3f} s per likelihood "
              f"(prior draws: {int((r < -1e29).sum())} failed, {int(((r < -1e5) & (r > -1e29)).sum())} under-sampled)")


def _prior_corners(out_dir: Path):
    """One corner plot per observable: the joint prior predictive distribution of the ten phi1-bin
    medians of replicated data (covered prior draws), with the observed bin medians marked."""
    import matplotlib

    matplotlib.use("Agg")
    import corner
    import matplotlib.pyplot as plt

    z = np.load(out_dir / "prior_predictive.npz")
    y, mask, b, phi1, g = z["y"], z["mask"], z["bin"], z["phi1"], z["covered"]
    y_rep = z["y_rep"][g]
    cen = [phi1[b == bb].mean() for bb in range(N_BINS)]
    tags = ["phi2", "parallax", "pm_ra", "pm_dec", "v_los"]
    for d in range(5):
        rep = np.stack([np.median(y_rep[:, (b == bb) & (mask[:, d] > 0), d], axis=1) for bb in range(N_BINS)], -1)
        obs = np.array([np.median(y[(b == bb) & (mask[:, d] > 0), d]) for bb in range(N_BINS)])
        pct = [float(np.mean(rep[:, bb] < obs[bb])) for bb in range(N_BINS)]
        rngs = []
        for bb in range(N_BINS):
            lo, hi = np.percentile(rep[:, bb], [1, 99])
            lo, hi = min(lo, obs[bb]), max(hi, obs[bb])
            rngs.append((lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)))
        labels = [f"bin {bb}\n$\\phi_1$={cen[bb]:+.1f}" for bb in range(N_BINS)]
        fig = corner.corner(rep, labels=labels, range=rngs, bins=25, smooth=1.0, truths=obs, truth_color="C3",
                            plot_datapoints=True, data_kwargs=dict(alpha=0.25, ms=2), levels=(0.393, 0.865),
                            color="C0", label_kwargs=dict(fontsize=8), max_n_ticks=3)
        axes = np.array(fig.axes).reshape(N_BINS, N_BINS)
        for bb in range(N_BINS):
            axes[bb, bb].set_title(f"obs {obs[bb]:.3g}\nP(rep<obs)={pct[bb]:.2f}", fontsize=8)
        fig.suptitle(f"Prior predictive: median {OBS_NAMES[d]} of replicated data in the 10 $\\phi_1$ bins "
                     f"({len(rep)} prior draws); red = observed bin medians", fontsize=13, y=1.01)
        fig.savefig(out_dir / f"prior_corner_{tags[d]}.png", dpi=75, bbox_inches="tight")
        plt.close(fig)
        print(f"prior_corner_{tags[d]}.png: P(rep < obs) per bin = {np.round(pct, 2)}")


def cmd_prior(args):
    """Prior predictive checks: draw from the prior, simulate, and ask whether the observed stream is a
    plausible draw -- per phi1 bin and observable for the bin mean and the bin scatter -- plus what the
    prior implies for the rotation curve, the stream coverage and the model's intrinsic widths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out) / "prior_predictive"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.replot:  # only redo the corner plots from the saved draws (no simulation, no GPU)
        return _prior_corners(out_dir)
    data = load_data()
    loglike, _ = make_loglike(data, n_release=args.n_release, n_steps=args.n_steps)
    y, err, mask, phi1, b = data["y"], data["err"], data["mask"], data["phi1"], data["bin"]
    n = args.n_ppc
    thetas = prior_sample(jr.key(4242), n)
    keys = jr.split(jr.key(99), n)

    def one(th, k):
        obs, ok, _ = simulate_stream(th, key=k, n_release=args.n_release, n_steps=args.n_steps)
        mu, tot, var, nb = loglike.predict(th, obs)
        host = host_potential(th[0], th[1], th[2])
        vc = jnp.stack([jnp.sqrt(R * host._gradient(jnp.array([R, 0.0, 0.0]), 0.0)[0]) * KMS for R in (8.122, 15.0, 20.0)])
        ext = jnp.percentile(obs[:, 0], jnp.array([1.0, 99.0]))
        return ok, mu, tot, var, nb, loglike.of(th, mu, tot, nb, jnp.asarray(y)), vc, ext, obs[::10]

    f = jax.jit(jax.vmap(one))
    t0 = time.time()
    chunks = [f(thetas[i:i + 32], keys[i:i + 32]) for i in range(0, n, 32)]
    ok, mu, tot, var, nb, ll, vc, ext, cloud = (np.concatenate([np.asarray(c[k]) for c in chunks]) for k in range(9))
    th = np.asarray(thetas)
    covered = ok & (nb.min(1) >= N_MIN_BIN)
    print(f"{n} prior draws simulated in {time.time() - t0:.0f}s: {int((~ok).sum())} failed, "
          f"{int((ok & ~covered).sum())} leave an observed bin under-sampled, {int(covered.sum())} fully covered")
    res = dict(n_draws=int(n), n_failed=int((~ok).sum()), n_undersampled=int((ok & ~covered).sum()), n_covered=int(covered.sum()))
    # what makes a draw fail to cover the observed stream?
    res["undersampled_fraction_by_t_end"] = {
        f"{lo}-{lo + 1} Gyr": float(np.mean(~covered[(th[:, 9] >= lo) & (th[:, 9] < lo + 1)])) for lo in (1, 2, 3, 4)}
    res["undersampled_bins_hist"] = np.bincount((nb[ok] < N_MIN_BIN).sum(1), minlength=N_BINS + 1).tolist()
    res["phi1_extent_1_99pct"] = dict(lead_median=float(np.median(ext[ok, 1])), trail_median=float(np.median(ext[ok, 0])),
                                      data=[float(phi1.min()), float(phi1.max())])
    for k, R in enumerate((8.122, 15.0, 20.0)):
        res[f"vc_{R}kpc_prior_pct_2.5_16_50_84_97.5"] = np.percentile(vc[:, k], [2.5, 16, 50, 84, 97.5]).round(1).tolist()
    # replicated data from the covered draws
    rng = np.random.default_rng(1)
    g = covered
    f_out = th[:, NAMES.index("f_out")]
    is_out = rng.random((n, len(phi1))) < f_out[:, None]
    y_rep = np.where(is_out[:, :, None], np.median(y, 0) + OUTLIER_WIDTH * (rng.random(mu.shape) - 0.5),
                     mu + np.sqrt(tot) * rng.standard_normal(mu.shape))
    # robust observed statistics (median, MAD scatter) so the three v_los outliers do not dominate
    mad = lambda a, ax=None: 1.4826 * np.nanmedian(np.abs(a - np.nanmedian(a, axis=ax, keepdims=True)), axis=ax)
    p_med, p_sc, p_track, obs_med, obs_sc = (np.full((N_BINS, 5), np.nan) for _ in range(5))
    for bb in range(N_BINS):
        for d in range(5):
            sel = (b == bb) & (mask[:, d] > 0)
            if sel.sum() < 3:
                continue
            rep = y_rep[g][:, sel, d]
            obs_med[bb, d], obs_sc[bb, d] = np.median(y[sel, d]), mad(y[sel, d])
            p_med[bb, d] = np.mean(np.median(rep, 1) < obs_med[bb, d])
            p_sc[bb, d] = np.mean(mad(rep, 1) < obs_sc[bb, d])
            p_track[bb, d] = np.mean(np.median(mu[g][:, sel, d], 1) < obs_med[bb, d])  # noise-free model track
    res.update(obs_names=OBS_NAMES, p_bin_median=p_med.tolist(), p_bin_scatter=p_sc.tolist(), p_bin_track_only=p_track.tolist(),
               n_bin_median_outside_95=int(np.sum((p_med < 0.025) | (p_med > 0.975))),
               n_bin_scatter_outside_95=int(np.sum((p_sc < 0.025) | (p_sc > 0.975))),
               n_bin_track_outside_95=int(np.sum((p_track < 0.025) | (p_track > 0.975))), n_bin_tests=int(np.isfinite(p_med).sum()))
    res["model_intrinsic_sigma_median"] = np.sqrt(np.median(var[g], axis=(0, 1))).tolist()
    res["model_intrinsic_sigma_pct_2.5_97.5"] = np.sqrt(np.percentile(np.median(var[g], 1), [2.5, 97.5], axis=0)).T.tolist()
    res["data_error_median"] = np.median(err, 0).tolist()
    res["observed_scatter_about_bin_median"] = np.nanmedian(obs_sc, 0).tolist()
    res["logL_prior_pct_2.5_50_97.5_max"] = [float(v) for v in (*np.percentile(ll[g], [2.5, 50, 97.5]), ll[g].max())]
    (out_dir / "prior_predictive.json").write_text(json.dumps(res, indent=1))
    np.savez(out_dir / "prior_predictive.npz", theta=th, ok=ok, covered=covered, mu=mu, tot=tot, var_model=var, bin_counts=nb,
             logL=ll, vc=vc, phi1_extent=ext, y_rep=y_rep, phi1=phi1, y=y, err=err, mask=mask, bin=b)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("p_bin")}, indent=1))

    o = np.argsort(phi1)
    fig, axs = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    cl = cloud[g].reshape(-1, 6)
    for d, ax in enumerate(axs):
        ax.plot(cl[:, 0], cl[:, d + 1], ",", color="0.8", alpha=0.3, rasterized=True, zorder=0)
        lo2, lo1, hi1, hi2 = np.percentile(mu[g][..., d], [2.5, 16, 84, 97.5], axis=0)
        ax.fill_between(phi1[o], lo2[o], hi2[o], color="C2", alpha=0.2, step="mid", label="prior: model track 95%")
        ax.fill_between(phi1[o], lo1[o], hi1[o], color="C2", alpha=0.4, step="mid", label="68%")
        r2 = np.nanpercentile(np.where(is_out[:, :, None], np.nan, y_rep)[g][..., d], [2.5, 97.5], axis=0)
        ax.plot(phi1[o], r2[0][o], color="C0", lw=0.8, ds="steps-mid", label="prior predictive (star-level) 95%")
        ax.plot(phi1[o], r2[1][o], color="C0", lw=0.8, ds="steps-mid")
        m = mask[:, d] > 0
        ax.errorbar(phi1[m], y[m, d], yerr=err[m, d], fmt="o", ms=3, color="k", lw=0.6, label="observed")
        lo, hi = np.nanpercentile(y_rep[g][..., d], [0.5, 99.5])
        ax.set_ylim(min(lo, np.percentile(y[m, d], 1)), max(hi, np.percentile(y[m, d], 99)))
        ax.set_ylabel(OBS_NAMES[d])
        if d == 0:
            ax.legend(fontsize=7, ncol=4, loc="upper center")
    axs[-1].set_xlim(phi1.min() - 1, phi1.max() + 1)
    axs[-1].set_xlabel(r"$\phi_1$ [deg]")
    fig.suptitle(f"Prior predictive check ({int(g.sum())} prior draws that cover the observed stream; grey: their particles)")
    fig.tight_layout()
    fig.savefig(out_dir / "prior_tracks.png", dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    for ax, pp, ttl in ((axs[0, 0], p_med, "P(replicated bin median < observed)"), (axs[0, 1], p_sc, "P(replicated bin scatter < observed)"),
                        (axs[0, 2], p_track, "P(model track < observed bin median)")):
        im = ax.imshow(pp.T, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(5), OBS_NAMES)
        ax.set_xlabel(r"$\phi_1$ bin")
        ax.set_title(ttl, fontsize=9)
        for (i_, j_), v in np.ndenumerate(pp):
            if np.isfinite(v):
                ax.text(i_, j_, f"{v:.2f}", ha="center", va="center", fontsize=6)
        plt.colorbar(im, ax=ax)
    ax = axs[0, 3]
    for k, R in enumerate((8.122, 15.0, 20.0)):
        ax.hist(vc[:, k], bins=40, range=(100, 450), histtype="step", lw=1.5, label=f"R = {R:g} kpc")
    ax.axvspan(220, 240, color="0.85", label="observed v_c(R0) ~ 220-240")
    ax.set_xlabel("circular velocity implied by the prior [km/s]")
    ax.legend(fontsize=7)
    ax = axs[1, 0]
    ax.scatter(th[:, 9], 10 ** th[:, 7] / 1e4, c=np.where(covered, "C0", "C3"), s=8)
    ax.set_yscale("log")
    ax.set_xlabel("t_end [Gyr]")
    ax.set_ylabel("initial progenitor mass [1e4 Msun]")
    ax.set_title("red: an observed bin is left under-sampled", fontsize=9)
    ax = axs[1, 1]
    ax.scatter(th[:, 0], th[:, 1], c=np.where(covered, "C0", "C3"), s=8)
    ax.set_xlabel("log10 M_s")
    ax.set_ylabel("r_s [kpc]")
    ax = axs[1, 2]
    ax.hist(np.clip(ll[g], -3000, None), bins=50, color="C0")
    ax.set_xlabel("log-likelihood of the data at prior draws (clipped at -3000)")
    ax = axs[1, 3]
    ax.hist(ext[ok, 0], bins=40, histtype="step", label="trailing end (1st pct)")
    ax.hist(ext[ok, 1], bins=40, histtype="step", label="leading end (99th pct)")
    ax.axvline(phi1.min(), color="k", ls="--")
    ax.axvline(phi1.max(), color="k", ls="--", label="observed extent")
    ax.set_xlabel(r"model stream extent in $\phi_1$ [deg]")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "prior_summary.png", dpi=120)
    plt.close(fig)
    _prior_corners(out_dir)
    print(f"wrote {out_dir}")


def _load_runs(out_dir: Path):
    runs = []
    for f in sorted(out_dir.glob("run*_final.npz")) + sorted(out_dir.glob("run*_ckpt.npz")):
        tag = f.name.split("_")[0]
        if any(r["tag"] == tag for r in runs):
            continue
        z = np.load(f)
        if "theta" in z.files:
            runs.append(dict(tag=tag, theta=z["theta"], logl=z["logl"], logl_birth=z["logl_birth"], finished=True,
                             n_evals=int(z["n_evals"])))
        else:  # unfinished run: dead points so far + its current live points
            runs.append(dict(tag=tag, theta=np.concatenate([z["dead_theta"], z["live_theta"]]),
                             logl=np.concatenate([z["dead_logl"], z["live_logl"]]),
                             logl_birth=np.concatenate([z["dead_logl_birth"], z["live_logl_birth"]]), finished=False,
                             n_evals=int(z["n_evals"])))
    return runs


def _wquant(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(q, c / c[-1], x[o])


def cmd_post(args):
    """Merge the runs -> evidence, posterior, derived quantities, corner plots, posterior predictive checks."""
    import matplotlib

    matplotlib.use("Agg")
    import corner
    import matplotlib.pyplot as plt

    out_dir = Path(args.out)
    if args.samples:  # weighted samples from elsewhere (e.g. an importance-sampled NPE posterior): theta, logl, weight
        out_dir.mkdir(parents=True, exist_ok=True)
        z = np.load(args.samples)
        keep = z["weight"] > 0
        theta, logl, w = z["theta"][keep], z["logl"][keep], z["weight"][keep] / z["weight"][keep].sum()
        lw, birth, run_id, runs = np.log(w), np.full(len(w), np.nan), np.zeros(len(w), int), [dict(tag="samples")]
        ess = 1.0 / np.sum(w**2)
        summary = dict(source=str(args.samples), n_particles=2 * args.n_release, n_steps=args.n_steps, x64=X64,
                       likelihood=args.likelihood, ess=float(ess), n_points=int(len(w)))
        print(f"weighted samples from {args.samples}: {len(w)} with weight > 0, ESS = {ess:.0f}")
    else:
        runs = _load_runs(out_dir)
        assert runs, f"no runs in {out_dir}"
        summary = dict(runs=[], n_particles=2 * args.n_release, n_steps=args.n_steps, x64=X64, likelihood=args.likelihood)
        for r in runs:
            _, lz = ns_weights(r["logl"], r["logl_birth"], jr.key(3))
            summary["runs"].append(dict(tag=r["tag"], finished=r["finished"], n_points=len(r["logl"]), n_evals=r["n_evals"],
                                        logZ=float(lz.mean()), logZ_err=float(lz.std()), max_logL=float(r["logl"].max())))
            print(f"{r['tag']}: finished {r['finished']}, {len(r['logl'])} points, {r['n_evals']} evals, "
                  f"logZ = {lz.mean():.2f} +- {lz.std():.2f}, max logL {r['logl'].max():.2f}")
        theta = np.concatenate([r["theta"] for r in runs])
        logl = np.concatenate([r["logl"] for r in runs])
        birth = np.concatenate([r["logl_birth"] for r in runs])
        run_id = np.concatenate([np.full(len(r["logl"]), i) for i, r in enumerate(runs)])
        lw, lz = ns_weights(logl, birth, jr.key(4))
        w = np.exp(lw - lw.max())
        w /= w.sum()
        ess = 1.0 / np.sum(w**2)
        summary.update(logZ=float(lz.mean()), logZ_err=float(lz.std()), ess=float(ess), n_points=int(len(w)),
                       n_evals=int(sum(r["n_evals"] for r in runs)), all_finished=bool(all(r["finished"] for r in runs)),
                       run_weight_share=[float(w[run_id == i].sum()) for i in range(len(runs))])
        print(f"merged: {len(w)} points, logZ = {lz.mean():.2f} +- {lz.std():.2f}, ESS = {ess:.0f}, "
              f"posterior mass per run {np.round(summary['run_weight_share'], 3)}")
    rng = np.random.default_rng(0)
    eq = theta[rng.choice(len(w), size=4000, p=w)]  # equal-weight posterior sample
    i_map = int(np.argmax(logl))

    # ---- derived: circular velocity of the fitted potential, enclosed dynamical mass -----------------
    def derived(th):
        host = host_potential(th[0], th[1], th[2])
        vc = lambda R: jnp.sqrt(R * host._gradient(jnp.array([R, 0.0, 0.0]), 0.0)[0]) * KMS
        return jnp.stack([vc(8.122), vc(15.0), vc(20.0)])

    sub = eq[:1000]
    dq = np.asarray(jax.jit(jax.vmap(derived))(jnp.asarray(sub)))
    g_kms = 4.300917270e-6  # kpc (km/s)^2 / Msun
    der = {"vc_8.122kpc [km/s]": dq[:, 0], "vc_15kpc [km/s]": dq[:, 1], "vc_20kpc [km/s]": dq[:, 2],
           "M_dyn(<15kpc)=R vc^2/G [1e11 Msun]": 15.0 * dq[:, 1] ** 2 / g_kms / 1e11,
           "M_dyn(<20kpc)=R vc^2/G [1e11 Msun]": 20.0 * dq[:, 2] ** 2 / g_kms / 1e11,
           "M_prog,0 [1e4 Msun]": 10 ** sub[:, NAMES.index("log10_m_prog")] / 1e4}
    table = {}
    lines = ["| parameter | prior | median | 16% | 84% | 2.5% | 97.5% | max-logL |", "|---|---|---|---|---|---|---|---|"]
    for k, (name, kind, a, b, _) in enumerate(PARAMS):
        q = _wquant(theta[:, k], w, [0.5, 0.16, 0.84, 0.025, 0.975])
        table[name] = dict(median=float(q[0]), lo=float(q[1]), hi=float(q[2]), lo95=float(q[3]), hi95=float(q[4]), maxL=float(theta[i_map, k]))
        prior = f"U[{a:g}, {b:g}]" if kind == "u" else f"N({a:g}, {b:g})"
        lines.append(f"| {name} | {prior} | {q[0]:.4g} | {q[1]:.4g} | {q[2]:.4g} | {q[3]:.4g} | {q[4]:.4g} | {theta[i_map, k]:.4g} |")
    for name, v in der.items():
        q = np.quantile(v, [0.5, 0.16, 0.84, 0.025, 0.975])
        table[name] = dict(median=float(q[0]), lo=float(q[1]), hi=float(q[2]), lo95=float(q[3]), hi95=float(q[4]))
        lines.append(f"| {name} | derived | {q[0]:.4g} | {q[1]:.4g} | {q[2]:.4g} | {q[3]:.4g} | {q[4]:.4g} | |")
    summary["posterior"] = table
    (out_dir / "posterior_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    np.savez(out_dir / "posterior.npz", names=np.array(NAMES), theta=theta, logw=lw, logl=logl, logl_birth=birth,
             run_id=run_id, equal_weight=eq, derived_names=np.array(list(der)), derived=np.stack(list(der.values()), -1),
             derived_theta=sub, prior_kind=np.array([p[1] for p in PARAMS]), prior_a=_A, prior_b=_B)

    # ---- corner plots (posterior vs prior range; per-run 1-d histograms as a convergence check) -------
    labels = [p[4] for p in PARAMS]
    for fname, idx in (("corner_potential_pal5.png", list(range(10))), ("corner_nuisance.png", list(range(10, NDIM))),
                       ("corner_all.png", list(range(NDIM)))):
        rngs = [((_A[k], _B[k]) if _KIND_U[k] else (_A[k] - 4 * _B[k], _A[k] + 4 * _B[k])) for k in idx]
        fig = corner.corner(eq[:, idx], labels=[labels[k] for k in idx], range=rngs, bins=30, smooth=1.0,
                            quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3g", plot_datapoints=False,
                            levels=(0.393, 0.865), color="C0", label_kwargs=dict(fontsize=9), title_kwargs=dict(fontsize=8))
        axes = np.array(fig.axes).reshape(len(idx), len(idx))
        for a_i, k in enumerate(idx):
            ax = axes[a_i, a_i]
            xx = np.linspace(*rngs[a_i], 200)
            pr = np.full_like(xx, 1 / (_B[k] - _A[k])) if _KIND_U[k] else np.exp(-0.5 * ((xx - _A[k]) / _B[k]) ** 2) / (_B[k] * np.sqrt(2 * np.pi))
            h = np.histogram(eq[:, k], bins=30, range=rngs[a_i])[0].max()
            ax.plot(xx, pr * len(eq) * (rngs[a_i][1] - rngs[a_i][0]) / 30, color="0.5", ls="--", lw=1)
            ax.set_ylim(0, 1.25 * h)
        fig.suptitle(f"Pal 5 + MW halo, galax Chen+24 spray, {'weighted samples' if args.samples else 'nested sampling'} (dashed: prior)", y=1.01, fontsize=11)
        fig.savefig(out_dir / fname, dpi=110, bbox_inches="tight")
        plt.close(fig)
    fig, axs = plt.subplots(4, 4, figsize=(17, 13))
    for k, ax in enumerate(axs.ravel()[:NDIM]):
        lo, hi = _wquant(theta[:, k], w, [0.001, 0.999])
        for i, r in enumerate(runs):
            m = run_id == i
            if w[m].sum() > 0:
                ax.hist(theta[m, k], bins=25, range=(lo, hi), weights=w[m] / w[m].sum(), histtype="step", lw=1, label=r["tag"])
        ax.hist(theta[:, k], bins=25, range=(lo, hi), weights=w, histtype="step", color="k", lw=2)
        ax.set_xlabel(labels[k])
    axs[0, 0].legend(fontsize=6)
    fig.suptitle("Marginals of the independent runs (colours) and of their merge (black)")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_runs.png", dpi=110)
    plt.close(fig)

    # ---- posterior predictive checks ----------------------------------------------------------------
    binned = args.likelihood == "bin"
    tracks = args.likelihood == "bspline"  # Gaussian stars (no outliers) scored through their B-spline tracks
    with_out = args.likelihood == "star"  # only the star likelihood has the outlier component
    stars = load_data()
    data = bin_data(stars) if binned else stars
    if tracks:
        loglike, lfo = make_loglike_bspline(data, bspline_ops_file(out_dir), n_release=args.n_release, n_steps=args.n_steps)
    else:
        loglike, lfo = make_loglike(data, n_release=args.n_release, n_steps=args.n_steps, binned=binned)
    n_ppc = args.n_ppc
    draws = jnp.asarray(eq[:n_ppc])
    keys = jr.split(jr.key(2026), n_ppc)  # a NEW spray realisation per draw: the predictive includes it

    def one(th, k):
        obs, ok, _ = simulate_stream(th, key=k, n_release=args.n_release, n_steps=args.n_steps)
        mu, tot, var, n = loglike.predict(th, obs)
        return obs, ok & (jnp.min(n) >= N_MIN_BIN), mu, tot, var, loglike.of(th, mu, tot, n, jnp.asarray(data["y"])), n

    f = jax.jit(jax.vmap(one))
    chunks = [f(draws[i:i + 20], keys[i:i + 20]) for i in range(0, n_ppc, 20)]
    obs, okp, mu, tot, var, ll_obs, nbin = (np.concatenate([np.asarray(c[k]) for c in chunks]) for k in range(7))
    good = okp
    print(f"PPC: {int(good.sum())}/{n_ppc} posterior draws give a fully sampled stream with a fresh spray key")
    y, err, mask, phi1, b = data["y"], data["err"], data["mask"], data["phi1"], data["bin"]
    f_out = eq[:n_ppc, NAMES.index("f_out")]
    is_out = (rng.random((n_ppc, len(phi1))) < f_out[:, None]) & with_out
    y_rep = mu + np.sqrt(tot) * rng.standard_normal(mu.shape)
    y_uni = np.median(y, 0)[None, None, :] + OUTLIER_WIDTH * (rng.random(mu.shape) - 0.5)
    y_rep = np.where(is_out[:, :, None], y_uni, y_rep)
    log_out = -(mask * np.log(OUTLIER_WIDTH)).sum(-1)

    def np_ll(th_f, mu_i, tot_i, yy):
        logn = (mask * (-0.5 * (yy - mu_i) ** 2 / tot_i - 0.5 * np.log(2 * np.pi * tot_i))).sum(-1)
        if binned:
            return logn.sum()
        return np.logaddexp(np.log1p(-th_f) + logn, np.log(th_f) + log_out).sum()

    if tracks:  # replicated stars -> the density of THEIR tracks, as for the observed ones
        ll_rep = np.asarray(jax.jit(jax.vmap(loglike.of))(draws, jnp.asarray(mu), jnp.asarray(tot), jnp.asarray(nbin), jnp.asarray(y_rep)))
    else:
        ll_rep = np.array([np_ll(f_out[i], mu[i], tot[i], y_rep[i]) for i in range(n_ppc)])
    ppc = dict(n_draws=int(good.sum()), p_loglike=float(np.mean(ll_rep[good] < ll_obs[good])),
               logL_obs_mean=float(ll_obs[good].mean()), logL_obs_std=float(ll_obs[good].std()),
               logL_rep_mean=float(ll_rep[good].mean()), logL_rep_std=float(ll_rep[good].std()))
    # outlier membership probability of each star (posterior mean)
    logn = (mask[None] * (-0.5 * (y[None] - mu) ** 2 / tot - 0.5 * np.log(2 * np.pi * tot))).sum(-1)
    p_out = 1 / (1 + (1 - f_out[:, None]) / f_out[:, None] * np.exp(logn - log_out[None]))
    p_out = p_out[good].mean(0) * with_out
    inl = p_out < 0.5
    ppc["n_outlier_stars"] = int((~inl).sum())
    # bin statistics: observed mean / std of each observable in each phi1 bin vs their replicated distribution
    zmean, zstd, pmean, pstd = (np.full((N_BINS, 5), np.nan) for _ in range(4))
    for bb in range(N_BINS):
        for d in range(5):
            sel = (b == bb) & (mask[:, d] > 0) & inl
            if binned and sel.sum() == 1:  # one observed value per cell: compare it with its replicates
                rm = y_rep[good][:, bb, d]
                zmean[bb, d], pmean[bb, d] = (y[bb, d] - rm.mean()) / rm.std(), np.mean(rm < y[bb, d])
                continue
            if sel.sum() < 3:
                continue
            keep = ~is_out[:, sel] & good[:, None]
            rep = np.where(keep, y_rep[:, sel, d], np.nan)[good]
            rm, rs = np.nanmean(rep, 1), np.nanstd(rep, 1)
            om, os_ = y[sel, d].mean(), y[sel, d].std()
            zmean[bb, d], zstd[bb, d] = (om - np.nanmean(rm)) / np.nanstd(rm), (os_ - np.nanmean(rs)) / np.nanstd(rs)
            pmean[bb, d], pstd[bb, d] = np.nanmean(rm < om), np.nanmean(rs < os_)
    ppc.update(z_bin_mean=zmean.tolist(), z_bin_std=zstd.tolist(), p_bin_mean=pmean.tolist(), p_bin_std=pstd.tolist(),
               frac_bin_mean_within_2sigma=float(np.mean(np.abs(zmean[np.isfinite(zmean)]) < 2)),
               frac_bin_std_within_2sigma=float(np.mean(np.abs(zstd[np.isfinite(zstd)]) < 2)) if np.isfinite(zstd).any() else None,
               obs_names=OBS_NAMES)
    # coverage of the per-star predictive intervals (inlier stars)
    zstar = (y[None] - mu) / np.sqrt(tot)
    zs = np.where((mask[None] > 0) & inl[None, :, None], zstar, np.nan)[good]
    cover = lambda d, c: float(np.mean(np.abs(zs[..., d][np.isfinite(zs[..., d])]) < c))  # masked entries excluded
    ppc["coverage_68"] = [cover(d, 1.0) for d in range(5)]
    ppc["coverage_95"] = [cover(d, 1.96) for d in range(5)]
    ppc["mean_pull"] = [float(np.nanmean(zs[..., d])) for d in range(5)]
    # stream density along phi1 -- NOT part of the likelihood
    cnt_obs = np.bincount(stars["bin"], minlength=N_BINS)
    frac = nbin[good] / nbin[good].sum(1, keepdims=True)
    cnt_rep = np.stack([rng.multinomial(len(stars["phi1"]), fr) for fr in frac])
    ppc["density_obs_counts"] = cnt_obs.tolist()
    ppc["density_rep_mean"] = cnt_rep.mean(0).tolist()
    ppc["density_rep_std"] = cnt_rep.std(0).tolist()
    ppc["density_z"] = ((cnt_obs - cnt_rep.mean(0)) / cnt_rep.std(0)).tolist()
    # likelihood scatter over the spray key at the max-likelihood point
    th_map = jnp.asarray(theta[i_map])
    ll_k = np.asarray(jax.jit(jax.vmap(lambda k: lfo(th_map, *simulate_stream(th_map, key=k, n_release=args.n_release, n_steps=args.n_steps)[:2])))(jr.split(jr.key(11), 20)))
    ll_k = ll_k[ll_k > -1e5]
    ppc.update(maxL_fixed_key=float(logl[i_map]), maxL_other_keys_mean=float(ll_k.mean()), maxL_other_keys_std=float(ll_k.std()))
    summary["ppc"] = ppc
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez(out_dir / "ppc.npz", theta=eq[:n_ppc], ok=good, mu=mu, tot=tot, var_model=var, y_rep=y_rep, is_outlier_rep=is_out,
             logL_obs=ll_obs, logL_rep=ll_rep, p_outlier_star=p_out, phi1=phi1, y=y, err=err, mask=mask, bin=b, edges=data["edges"],
             bin_counts_model=nbin, particles=obs[:, ::20])
    print(json.dumps({k: v for k, v in ppc.items() if not isinstance(v, list)}, indent=1))

    # ---- PPC figures --------------------------------------------------------------------------------
    phi1_cells = phi1
    cloud = obs[good][:, ::25].reshape(-1, 6)
    fig, axs = plt.subplots(5, 2, figsize=(15, 16), gridspec_kw=dict(width_ratios=[3, 1]))
    for d in range(5):
        ax = axs[d, 0]
        phi1 = phi1_cells if phi1_cells.ndim == 1 else phi1_cells[:, d]
        o = np.argsort(phi1)
        if binned:
            sm = stars["mask"][:, d] > 0
            ax.plot(stars["phi1"][sm], stars["y"][sm, d], ".", color="0.35", ms=3, zorder=1, label="observed stars (not fitted directly)")
        ax.plot(cloud[:, 0], cloud[:, d + 1], ",", color="0.75", alpha=0.5, zorder=0, rasterized=True)
        lo2, lo1, med, hi1, hi2 = np.nanpercentile(np.where(is_out[:, :, None], np.nan, y_rep)[good][..., d], [2.5, 16, 50, 84, 97.5], axis=0)
        ax.fill_between(phi1[o], lo2[o], hi2[o], color="C0", alpha=0.18, step="mid", label="posterior predictive 95%")
        ax.fill_between(phi1[o], lo1[o], hi1[o], color="C0", alpha=0.30, step="mid", label="68%")
        mlo, mhi = np.percentile(mu[good][..., d], [16, 84], axis=0)
        ax.fill_between(phi1[o], mlo[o], mhi[o], color="C3", alpha=0.7, step="mid", label="model track (68%)")
        m = mask[:, d] > 0
        ax.errorbar(phi1[m & inl], y[m & inl, d], yerr=err[m & inl, d], fmt="o", ms=3, color="k", lw=0.7, label="observed")
        ax.plot(phi1[m & ~inl], y[m & ~inl, d], "x", color="C1", ms=6, label="flagged outlier (P>0.5)")
        ok_y = y[m & inl, d]
        pad = 4 * np.std(ok_y)
        ax.set_ylim(np.median(ok_y) - pad, np.median(ok_y) + pad)
        ax.set_xlim(stars["phi1"].min() - 1, stars["phi1"].max() + 1)
        ax.set_ylabel(OBS_NAMES[d])
        for e in data["edges"]:
            ax.axvline(e, color="0.85", lw=0.5, zorder=-1)
        if d == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax2 = axs[d, 1]
        zz = np.nanmean(zs[..., d], axis=0)
        ax2.hist(zz[np.isfinite(zz)], bins=20, range=(-4, 4), density=True, color="C0", alpha=0.6)
        xx = np.linspace(-4, 4, 200)
        ax2.plot(xx, np.exp(-0.5 * xx**2) / np.sqrt(2 * np.pi), "k--", lw=1)
        ax2.set_xlabel(f"pull (obs - model)/sigma_tot; 68% cov. {ppc['coverage_68'][d]:.2f}")
    axs[-1, 0].set_xlabel(r"$\phi_1$ [deg] (Pal5PriceWhelan18 frame)")
    fig.suptitle("Posterior predictive check: Pal 5 stream observables (grey: mock-stream particles of posterior draws)")
    fig.tight_layout()
    fig.savefig(out_dir / "ppc_tracks.png", dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, zz, ttl in ((axs[0], zmean, "bin MEAN: z of observed vs replicated"), (axs[1], zstd, "bin STD: z of observed vs replicated")):
        im = ax.imshow(zz.T, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        ax.set_yticks(range(5), OBS_NAMES)
        ax.set_xlabel(r"$\phi_1$ bin")
        ax.set_title(ttl)
        for (i_, j_), v in np.ndenumerate(zz):
            if np.isfinite(v):
                ax.text(i_, j_, f"{v:+.1f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax)
    ax = axs[2]
    cen = 0.5 * (data["edges"][1:] + data["edges"][:-1])
    ax.errorbar(cen, cnt_rep.mean(0), yerr=cnt_rep.std(0), fmt="s", color="C0", label="model (posterior predictive), not fitted")
    ax.plot(cen, cnt_obs, "ko", label="observed stars per bin (equal-count bins)")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel("stars per bin")
    ax.set_title("Density along the stream (no selection function)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "ppc_bin_stats.png", dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].hist(ll_rep[good], bins=30, color="C0", alpha=0.6, label="replicated data")
    axs[0].hist(ll_obs[good], bins=30, color="k", histtype="step", lw=1.5, label="observed data")
    axs[0].set_xlabel("log-likelihood under the posterior draw")
    axs[0].set_title(f"PPC p-value P(logL_rep < logL_obs) = {ppc['p_loglike']:.2f}")
    axs[0].legend()
    axs[1].plot(cloud[:, 0], cloud[:, 1], ",", color="0.6", rasterized=True)
    axs[1].plot(stars["phi1"], stars["y"][:, 0], "k.", ms=4)
    axs[1].set_xlabel(r"$\phi_1$ [deg]")
    axs[1].set_ylabel(r"$\phi_2$ [deg]")
    axs[1].set_xlim(-30, 30)
    axs[1].set_ylim(-4, 8)
    axs[1].set_title("Sky: posterior mock streams (grey) and observed stars")
    fig.tight_layout()
    fig.savefig(out_dir / "ppc_loglike_sky.png", dpi=120)
    plt.close(fig)
    print(f"wrote results to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("bench", "run", "worker", "post", "selftest", "prior"):
        sp = sub.add_parser(name)
        sp.add_argument("--out", default=str(OUT))
        sp.add_argument("--cpu", action="store_true")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--gpus", type=int, default=8, help="run: number of free GPUs = independent runs")
        sp.add_argument("--nlive", type=int, default=80, help="live points PER RUN")
        sp.add_argument("--num-delete", type=int, default=40, help="points replaced per iteration = likelihood batch")
        sp.add_argument("--num-inner", type=int, default=0, help="slice steps per new point (0 -> 2 * ndim)")
        sp.add_argument("--n-release", type=int, default=N_RELEASE)
        sp.add_argument("--n-steps", type=int, default=N_STEPS)
        sp.add_argument("--max-hours", type=float, default=1e9)
        sp.add_argument("--replot", action="store_true", help="prior: redo the corner plots from the saved draws")
        sp.add_argument("--n-ppc", type=int, default=200, help="post: posterior draws for the predictive checks")
        sp.add_argument("--samples", default="", help="post: npz with theta, logl, weight to use instead of the nested-sampling runs")
        sp.add_argument("--likelihood", choices=("star", "bin", "bspline"), default="star",
                        help="star: each observed star vs the model bin track; bin: observed bin centre vs model bin centre; "
                             "bspline: B-spline tracks of the observed stars vs those of Gaussian stars about the bin track")
        sp.add_argument("--rotation-curve", choices=("",) + tuple(ROT_CURVES), default="",
                        help="run: also fit this rotation curve (Gaussian term added to the stream log-likelihood)")
        sp.add_argument("--gpu-ids", default="", help="run: comma-separated GPU indices / UUIDs to use instead of autocvd")
    args = ap.parse_args()
    {"bench": cmd_bench, "run": cmd_run, "worker": cmd_worker, "post": cmd_post, "selftest": cmd_selftest, "prior": cmd_prior}[args.mode](args)


if __name__ == "__main__":
    main()
