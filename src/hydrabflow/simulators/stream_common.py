"""Shared helpers for the stellar-stream simulators (agama, gala, ...).

Ports the reference project's ``utils/utils_simulate.py`` (prior sampling over global potential
parameters + per-stream local phase-space parameters, and the Galactocentric -> ICRS sky
projection) in a simulator-agnostic form. Every stream simulator draws its parameters through
:func:`sample_stream_prior` / :func:`sample_stream_prior_shared_global` and projects its output
through :func:`sky_projection`.

Prior specs are plain mappings ``{type: uniform|normal|identity, prior_parameters: [...]}`` —
the same format the reference project used, so its configs translate 1:1. ``identity`` entries
are fixed constants (fed to the forward model but not inferred); the inferred parameter lists
are derived as "every non-identity entry".
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

# Observed Milky Way circular-velocity curve (Zhou et al. 2023 table, as used and labeled by
# the reference project): radii [kpc], Vc [km/s], and its 1-sigma uncertainty [km/s]. The
# simulators evaluate their model rotation curve on OBS_R_KPC; the noise augmentation uses
# OBS_SIGMA_VC; the optional ``vcirc_rejection`` prior cut compares against OBS_VC_KMS.
OBS_R_KPC = np.array([
    5.24, 5.74, 6.25, 6.77, 7.23, 7.83, 8.21, 8.78, 9.26, 9.75,
    10.25, 10.75, 11.25, 11.75, 12.24, 12.74, 13.25, 13.74, 14.23, 14.74,
    15.23, 15.74, 16.24, 16.74, 17.23, 17.74, 18.35, 18.90, 19.50, 20.41,
    21.28, 22.39, 23.16, 24.00,
])
OBS_VC_KMS = np.array([
    225.10, 233.53, 234.30, 233.17, 236.19, 236.00, 233.19, 233.15, 232.15, 231.24,
    230.34, 230.54, 229.11, 227.48, 226.69, 225.56, 224.90, 223.57, 221.10, 220.19,
    219.59, 217.36, 216.61, 217.28, 216.25, 213.81, 217.53, 212.10, 210.46, 206.69,
    207.71, 203.72, 205.20, 200.64,
])
OBS_SIGMA_VC = np.array([
    0.69, 0.68, 0.62, 0.60, 0.45, 0.29, 0.26, 0.22, 0.17, 0.16,
    0.17, 0.18, 0.19, 0.20, 0.25, 0.27, 0.27, 0.31, 0.40, 0.43,
    0.50, 0.68, 0.74, 0.87, 1.02, 1.15, 1.45, 1.58, 1.32, 1.71,
    1.69, 2.01, 2.50, 4.94,
])

# Huang et al. (2016) rotation curve out to ~100 kpc (three tracer samples — HI, PRCG, HKG —
# combined for full radial coverage), used to probe / extend the observational space beyond the
# Zhou (2023) range (~24 kpc). Columns: radii [kpc], Vc [km/s], 1-sigma [km/s].
_HUANG = np.array([
    [4.60, 231.24, 7.00], [5.08, 230.46, 7.00], [5.58, 230.01, 7.00],
    [6.10, 239.61, 7.00], [6.57, 246.27, 7.00], [7.07, 243.49, 7.00],
    [7.58, 242.71, 7.00], [8.04, 243.23, 7.00],
    [8.34, 239.89, 5.92], [8.65, 237.26, 6.29], [9.20, 235.30, 5.60],
    [9.62, 230.99, 5.49], [10.09, 228.41, 5.62], [10.58, 224.26, 5.87],
    [11.09, 224.94, 7.02], [11.58, 233.57, 7.65], [12.07, 240.02, 6.17],
    [12.73, 242.21, 8.64], [13.72, 261.78, 14.89], [14.95, 259.26, 30.84],
    [15.52, 268.57, 49.67], [16.55, 261.17, 50.91], [17.56, 240.66, 49.91],
    [18.54, 215.31, 24.80], [19.50, 214.99, 24.42], [21.25, 251.68, 19.50],
    [23.78, 259.65, 19.62], [26.22, 242.02, 18.66], [28.71, 224.11, 16.97],
    [31.29, 211.20, 16.43], [33.73, 217.93, 17.66], [36.19, 219.33, 18.44],
    [38.73, 213.31, 17.29], [41.25, 200.05, 17.72], [43.93, 190.15, 18.65],
    [46.43, 198.95, 20.70], [48.71, 192.91, 19.24], [51.56, 198.90, 21.74],
    [57.03, 185.88, 21.56], [62.55, 173.89, 22.87], [69.47, 196.36, 25.89],
    [79.27, 175.05, 22.71], [98.97, 147.72, 23.55],
])
HUANG_R_KPC, HUANG_VC_KMS, HUANG_SIGMA_VC = _HUANG.T


# --------------------------------------------------------------------------------------------- #
# Ibata et al. (2023, arXiv:2311.17202) ancillary potential observables.
#
# Three deterministic functions of the AGAMA Potential (no stream simulation needed): the HI
# terminal-velocity curve v_term(l), the local vertical force -> surface density Sigma(z=1.1 kpc),
# and the vertical stellar-density profile rho(z) at R0. See conf/simulator (Ibata potential) and
# new_constrains.md. Unit system is agama's (Msun, kpc, km/s); G below must match agama.G (verified
# to ~1e-10 relative). Fix R0 = 8.178 kpc (GRAVITY 2019), as Ibata does.
# --------------------------------------------------------------------------------------------- #
R0_KPC = 8.178                      # Solar galactocentric radius, fixed (GRAVITY 2019)
G_KPC_KMS2_MSUN = 4.300917270e-6    # Newton's G in (kpc, km/s, Msun); == agama.G

# Fixed grids the ancillary observables are evaluated on (single source of truth; NOT stored in
# the npz — only the per-row observable VALUES are, exactly as for the vcirc_kms rotation curve).
# Terminal velocity: the first-quadrant inner-Galaxy longitudes of the OBSERVED curve
# (McClure-Griffiths & Dickey 2016, l = 31..67 deg in 2 deg bins; == assets/terminal_velocity.csv
# `l_deg`), so the model curve aligns bin-for-bin with the data at real-data inference time. All
# satisfy |sin l| > 0.5 (tangent-point method; also avoids the bar). The 4th-quadrant curve
# (McClure-Griffiths & Dickey 2007) is not on VizieR; add it here + to the CSV if obtained.
VTERM_L_DEG = np.arange(31.0, 67.0 + 1e-9, 2.0)
# Vertical stellar-density profile: z = 0.1 -> 5 kpc (shape only; normalization marginalized).
RHO_Z_KPC = np.linspace(0.1, 5.0, 20)

# Reported observational uncertainties feeding the resampling of the SIMULATED observables
# (the observed central values enter only at real-data inference time).
VTERM_SIGMA_KMS = 6.2               # per (2-deg-averaged) terminal-velocity point (Ibata 2023)
SIGMA_Z_Z_KPC = 1.1                 # height of the surface-density datum
SIGMA_Z_OBS_MSUN_PC2 = 71.0         # Sigma(1.1 kpc) observed (Kuijken & Gilmore 1991)
SIGMA_Z_ERR_MSUN_PC2 = 6.0         # its 1-sigma
RHO_Z_REL_ERR = 0.1                 # assumed per-point relative uncertainty for rho(z) resampling

# Observed central values, entering ONLY at real-data inference time (simulated observables carry
# their own per-row model values). v_term: HI terminal velocity, McClure-Griffiths & Dickey (2016),
# on the VTERM_L_DEG grid (== assets/terminal_velocity.csv `vterm_kms`). Sigma(1.1 kpc) reuses
# SIGMA_Z_OBS_MSUN_PC2 above. (rho(z) observed data is a documented TODO — no constant here yet.)
OBS_VTERM_KMS = np.array([
    113.67, 107.16, 101.03, 93.33, 89.82, 79.18, 75.71, 71.22, 70.21, 69.24,
    68.42, 64.28, 50.48, 45.97, 42.65, 38.55, 34.12, 27.52, 21.87,
])
assert OBS_VTERM_KMS.shape == VTERM_L_DEG.shape, "OBS_VTERM_KMS must align with VTERM_L_DEG"


def vcirc_from_potential(pot, R) -> np.ndarray:
    """Circular velocity vc(R) [km/s] in the midplane; vc^2 = R dPhi/dR = -R F_R. NaN where <0."""
    R = np.atleast_1d(np.asarray(R, float))
    xyz = np.column_stack([R, np.zeros_like(R), np.zeros_like(R)])
    fR = pot.force(xyz)[:, 0]
    v2 = -R * fR
    return np.sqrt(np.where(v2 > 0, v2, np.nan))


def terminal_velocity(pot, l_deg=VTERM_L_DEG, R0: float = R0_KPC) -> np.ndarray:
    """HI terminal velocity v_term(l) (Ibata 2023, Eq. 13), tangent-point method.

    ``v_term(l) = sgn(sin l) * vc(R0 |sin l|) - vc(R0) * sin l``. Keep to |sin l| > 0.5 (the
    default ``VTERM_L_DEG`` grid already enforces this). Signed (not log) — terminal velocities
    change sign between quadrants.
    """
    l = np.radians(np.atleast_1d(np.asarray(l_deg, float)))
    sinl = np.sin(l)
    Rt = R0 * np.abs(sinl)
    return np.sign(sinl) * vcirc_from_potential(pot, Rt) - vcirc_from_potential(pot, R0)[0] * sinl


def surface_density(pot, z: float = SIGMA_Z_Z_KPC, R: float = R0_KPC) -> float:
    """Local surface density Sigma(z) [Msun/pc^2] from the vertical force (Kuijken & Gilmore 1991).

    Plane-parallel slab: ``Sigma(z) = |K_z| / (2 pi G)`` with ``K_z = -dPhi/dz = F_z``. Returns a
    scalar; matches Ibata's Sigma(1.1 kpc) = 71 +/- 6 Msun/pc^2 datum.
    """
    Fz = float(np.asarray(pot.force([R, 0.0, z])).reshape(-1)[2])
    Kz = -Fz
    return Kz / (2.0 * np.pi * G_KPC_KMS2_MSUN) / 1.0e6


def vertical_density_profile(disk_pot, z_vals=RHO_Z_KPC, R: float = R0_KPC) -> np.ndarray:
    """Vertical stellar-density profile rho(z) [Msun/kpc^3] at R (Ibata et al. 2017b).

    SHAPE only (free multiplicative normalization). ``disk_pot`` must be the STELLAR-disk
    component(s) alone (thin + thick), so gas/halo do not contaminate the stellar profile.
    """
    z_vals = np.atleast_1d(np.asarray(z_vals, float))
    xyz = np.column_stack([np.full_like(z_vals, R), np.zeros_like(z_vals), z_vals])
    return np.asarray(disk_pot.density(xyz), dtype=float)


def extended_rotation_curve(split_kpc: float | None = None):
    """Union rotation-curve grid: Zhou (2023) up to ``split_kpc``, Huang (2016) beyond it.

    Returns ``(r_kpc, vc_kms, sigma_kms)`` — the radii the model curve is evaluated on and the
    observed reference (Zhou below the split, Huang above), sorted by radius. ``split_kpc``
    defaults to the largest Zhou radius, so the Zhou grid is kept intact and only the
    larger-radius Huang points are appended.
    """
    split = float(OBS_R_KPC.max()) if split_kpc is None else float(split_kpc)
    hi = HUANG_R_KPC > split
    r = np.concatenate([OBS_R_KPC, HUANG_R_KPC[hi]])
    vc = np.concatenate([OBS_VC_KMS, HUANG_VC_KMS[hi]])
    sig = np.concatenate([OBS_SIGMA_VC, HUANG_SIGMA_VC[hi]])
    order = np.argsort(r)
    return r[order], vc[order], sig[order]


def sample_prior_value(spec: Mapping, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``(n, 1)`` samples from one prior spec (uniform / normal / identity)."""
    kind = spec["type"]
    p = list(spec["prior_parameters"])
    if kind == "uniform":
        return rng.uniform(p[0], p[1], size=(n, 1))
    if kind == "normal":
        return rng.normal(p[0], p[1], size=(n, 1))
    if kind == "identity":
        return np.full((n, 1), float(p[0]))
    raise ValueError(f"Unknown prior type '{kind}' (expected uniform|normal|identity)")


def inferred_names(priors: Mapping[str, Mapping]) -> list[str]:
    """Names of the non-fixed (inferred) entries of a prior-spec mapping, in config order."""
    return [k for k, spec in priors.items() if spec["type"] != "identity"]


def sample_stream_prior(
    priors_global: Mapping[str, Mapping],
    priors_local: Mapping[str, Mapping[str, Mapping]],
    target_streams: Mapping[str, int],
    n: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Single-stream draw: global parameters, a random stream index ``j``, and that stream's
    local parameters. Every returned array has shape ``(n, 1)``."""
    out: Dict[str, np.ndarray] = {}
    for key, spec in priors_global.items():
        out[key] = sample_prior_value(spec, n, rng)

    stream_by_j = {j: name for name, j in target_streams.items()}
    js = rng.choice(sorted(stream_by_j), size=n)
    out["j"] = js.reshape(n, 1).astype(np.float64)

    local_keys = list(next(iter(priors_local.values())).keys())
    for key in local_keys:
        col = np.empty((n, 1))
        for j, name in stream_by_j.items():
            mask = js == j
            col[mask] = sample_prior_value(priors_local[name][key], int(mask.sum()), rng)
        out[key] = col
    return out


def sample_stream_prior_shared_global(
    priors_global: Mapping[str, Mapping],
    priors_local: Mapping[str, Mapping[str, Mapping]],
    target_streams: Mapping[str, int],
    n: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Compositional draw: one global draw shared by *all* streams of each dataset.

    Globals come back ``(n, 1)``; ``j`` and each local parameter ``(n, n_streams, 1)``, with the
    stream axis ordered by the ``j`` index.
    """
    out: Dict[str, np.ndarray] = {}
    for key, spec in priors_global.items():
        out[key] = sample_prior_value(spec, n, rng)

    stream_by_j = {j: name for name, j in target_streams.items()}
    js = np.array(sorted(stream_by_j))
    m = len(js)
    out["j"] = np.tile(js.reshape(1, m, 1), (n, 1, 1)).astype(np.float64)

    local_keys = list(next(iter(priors_local.values())).keys())
    for key in local_keys:
        col = np.empty((n, m, 1))
        for idx, j in enumerate(js):
            col[:, idx] = sample_prior_value(priors_local[stream_by_j[j]][key], n, rng)
        out[key] = col
    return out


def sky_projection(sim_data_cartesian: np.ndarray) -> np.ndarray:
    """Project Galactocentric phase-space coordinates to observed ICRS quantities.

    Input ``(n, n_particles, 6)`` = (x, y, z [kpc], vx, vy, vz [km/s]); output same shape with
    (ra, dec [deg], distance [kpc], pm_ra_cosdec, pm_dec [mas/yr], v_los [km/s]).
    """
    import astropy.units as u
    from astropy.coordinates import ICRS, Galactocentric

    n, n_particles = sim_data_cartesian.shape[0], sim_data_cartesian.shape[1]
    flat = sim_data_cartesian.reshape(-1, 6)
    gc = Galactocentric(
        x=flat[:, 0] * u.kpc, y=flat[:, 1] * u.kpc, z=flat[:, 2] * u.kpc,
        v_x=flat[:, 3] * u.km / u.s, v_y=flat[:, 4] * u.km / u.s, v_z=flat[:, 5] * u.km / u.s,
    )
    c = gc.transform_to(ICRS())
    projected = np.stack(
        [
            c.ra.to(u.deg).value,
            c.dec.to(u.deg).value,
            c.distance.to(u.kpc).value,
            c.pm_ra_cosdec.to(u.mas / u.yr).value,
            c.pm_dec.to(u.mas / u.yr).value,
            c.radial_velocity.to(u.km / u.s).value,
        ],
        axis=-1,
    )
    return projected.reshape(n, n_particles, 6)
