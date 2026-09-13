"""Stellar-stream forward model built on AGAMA (CPU, parallelized with joblib).

Ports the reference project's ``utils/utils_agama_simulator.py`` (particle-spray stream
generation in a bulge + two-power triaxial halo + exponential disk potential, Fardal+2015
seeding, progenitor potential included) and ``agama_vcirc_worker.py`` (model circular-velocity
curve on the observed radii grid) into a registered :class:`BaseSimulator`.

Hierarchical structure:
  * **global** parameters: the Milky Way potential (halo ``rho/gamma/a/q``, disk
    ``r/z/Sigma``) — shared by every stream evolved in that potential;
  * **local** parameters: each stream's present-day phase-space coordinates (``vr``, ``r``,
    ``mu_ra_cosdec``, ``mu_dec``) — with fixed (identity-prior) progenitor mass/size, age and
    sky position per stream;
  * **context**: the stream index ``j`` (which of the target streams an observation is).

Observables:
  * ``sim_data_projected`` ``(n, n_particles, 6)`` — (ra, dec, distance, pm_ra_cosdec, pm_dec,
    v_los) of the stream particles (what the summary network sees);
  * ``vcirc_kms`` ``(n, n_radii, 1)`` — the model rotation curve on the observed radii grid
    (the fusion network's second input);
  * ``sim_data_carthesian`` is also stored for traceability/NaN masks but is not an observable.

AGAMA runs on CPU only, so simulation batches are farmed out with ``joblib`` (one process per
stream realization) — this is the CPU counterpart of the GPU ``jax.vmap`` path other stream
simulators (odisseo/galax) use. ``agama`` is imported lazily inside the workers so the package
imports fine in config-only contexts.

Config: see ``conf/simulator/stream_agama.yaml`` (``simulator.params``: ``priors_global``,
``priors_local``, ``target_streams``, ``n_particles``, ``n_workers``).
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.registry import register_simulator
from hydrabflow.simulators.stream_common import (
    OBS_R_KPC,
    OBS_SIGMA_VC,
    OBS_VC_KMS,
    R0_KPC,
    RHO_Z_KPC,
    VTERM_L_DEG,
    convert_concentration,
    extended_rotation_curve,
    inferred_names,
    sample_stream_prior,
    sample_stream_prior_shared_global,
    sky_projection,
    surface_density,
    terminal_velocity,
    vcirc_from_potential,
    vertical_density_profile,
)
from hydrabflow.utils.progress import joblib_row_progress
from hydrabflow.utils.quiet import quiet_worker

# Fixed bulge component of the host potential (not inferred), from the reference project.
BULGE_PARAMS = dict(
    type="Spheroid",
    scaleRadius=75 / 1e3,
    densityNorm=9.6e10,
    gamma=0,
    alpha=1,
    beta=1.8,
    cutoffStrength=2,
    outerCutoffRadius=2.1,
    axisRatioY=1.0,
    axisRatioZ=0.5,
)

# Fixed atomic (HI) and molecular (H2) gas disks, McMillan (2017) best-fit params (see
# new_constrains.md). Positive scaleHeight => isothermal sech^2 vertical profile (McMillan's gas
# disks); innerCutoffRadius is the central hole R_m (Sigma ~ exp(-R_m/R - R/R_d)). Added only when
# the potential config enables gas disks (the Ibata model) — needed for the inner-Galaxy terminal
# velocity; absent from the legacy potential.
GAS_HI_PARAMS = dict(
    type="Disk", surfaceDensity=5.31e7, scaleRadius=7.0, innerCutoffRadius=4.0, scaleHeight=0.085
)
GAS_H2_PARAMS = dict(
    type="Disk", surfaceDensity=2.18e9, scaleRadius=1.5, innerCutoffRadius=12.0, scaleHeight=0.045
)


def _agama():
    """Import agama with the (kpc, km/s, Msun) unit system set. Safe to call repeatedly."""
    import agama

    agama.setUnits(length=1, velocity=1, mass=1)
    return agama


# Default potential configuration = the legacy (pre-Ibata) model: untruncated halo, no gas disks,
# no thick disk, isothermal (sech^2) disk vertical profile. The Ibata model overrides these via
# ``simulator.params`` (see ``AgamaStreamSimulator._pot_cfg``). Kept as the ``pot_cfg=None`` default
# so existing configs / subclasses (stream_agama_rnbody) build exactly the same potential as before.
_DEFAULT_POT_CFG = dict(
    halo_r_t_kpc=float("inf"),
    gas_disks=False,
    thick_disk=False,
    disk_vertical="isothermal",  # "isothermal" (sech^2, +h) | "exponential" (exp(-|z|/h), -h)
    bulge_density_norm=BULGE_PARAMS["densityNorm"],  # override the fixed bulge densityNorm
    halo_parameterization="rho_a",  # "rho_a" (densityNorm + scaleRadius) | "m200_c" (M200 + c_v')
    halo_H0_kms_mpc=70.4,           # McMillan (2017) cosmology, used only by the m200_c conversion
    halo_Delta_mass=200.0,          # M200 overdensity (x rho_crit)
    halo_Delta_c=94.0,              # c_v' concentration overdensity (x rho_crit, McMillan)
)


def _resolve_pot_cfg(pot_cfg: Mapping | None) -> dict:
    cfg = dict(_DEFAULT_POT_CFG)
    if pot_cfg:
        cfg.update(pot_cfg)
    return cfg


def _thin_disk_params(p: Mapping[str, float], hsign: float) -> dict:
    return dict(
        type="Disk",
        scaleRadius=p["r_Disk"],
        scaleHeight=hsign * p["z_Disk"],
        surfaceDensity=p["Sigma_Disk"],
        sersicIndex=1,
        innerCutoffRadius=0,
    )


def _thick_disk_params(p: Mapping[str, float], hsign: float) -> dict:
    # Reparametrized coupling zd_thick > zd_thin: thick scale height = z_Disk + dz_thick_Disk
    # (dz_thick_Disk > 0), so no rejection is needed to enforce Ibata's zd_thick > zd_thin.
    return dict(
        type="Disk",
        scaleRadius=p["r_thick_Disk"],
        scaleHeight=hsign * (p["z_Disk"] + p["dz_thick_Disk"]),
        surfaceDensity=p["Sigma_thick_Disk"],
        sersicIndex=1,
        innerCutoffRadius=0,
    )


def _halo_shape_extras(p: Mapping[str, float]) -> dict:
    """Optional halo shape/sharpness parameters, each defaulting to the historical fixed value so a
    config that does not declare them reproduces the previous potential exactly.

    * ``alpha_TwoPowerTriaxial_halo`` — the two-power transition sharpness, previously hardcoded to
      1. It is the honest way to ask for an inner-steep / outer-declining halo: at fixed (M200, c)
      the only knob that could do that was ``gamma``, which is why gamma's posterior piles up at its
      prior boundary. With gamma = 1 and alpha = 2 the halo reaches v_c ~ 183 km/s at R = 4 kpc and
      still declines outwards, which is the shape the terminal-velocity and outer-rotation-curve
      data are jointly asking for.
    * ``p_TwoPowerTriaxial_halo`` — the intermediate axis ratio (``axisRatioY``), previously
      hardcoded to 1, i.e. the halo was forced axisymmetric.
    * ``tilt_TwoPowerTriaxial_halo`` — inclination [deg] of the halo's symmetry axis relative to the
      disc, passed as agama's Euler ``orientation=(0, radians(tilt), 0)`` (a rotation about the
      Sun-centre x axis; agama takes radians).

    The triaxiality/tilt pair is the literature-supported direction for the *inner* halo: Nibauer &
    Bonaca (2025) infer axis ratios 1 : 0.75 : 0.70 with the major axis tilted 18-20 deg out of the
    plane at r ~ 12 kpc from GD-1; Woudenberg & Helmi (2024) measure inner-halo triaxiality
    (p = 1.013 +- 0.006, q = 1.204) inside 20 kpc; and tilt is the generic expectation in
    simulations (Emami et al. 2021; Auriga's median disc-halo misalignment 19 +- 20 deg). A
    radius-dependent q(r) was considered and rejected: the published radial variation sets in at
    r >~ 30-150 kpc, outside these streams' 5-30 kpc range, and for a 10^12 Msun halo simulations
    find axis ratios almost independent of radius inside 30 kpc (Chua et al. 2019; Shao et al. 2021).
    """
    extras: dict = {"alpha": float(p.get("alpha_TwoPowerTriaxial_halo", 1.0))}
    extras["axisRatioY"] = float(p.get("p_TwoPowerTriaxial_halo", 1.0))
    tilt = float(p.get("tilt_TwoPowerTriaxial_halo", 0.0))
    if tilt != 0.0:
        # agama's Euler angles are in RADIANS (verified: orientation=(0, pi/2, 0) swaps the y and z
        # densities exactly); the config declares the tilt in degrees, so convert here. The second
        # Euler angle is a rotation about the x axis (the Sun-centre line), tipping the halo's
        # minor axis out of the disc normal toward the direction of Galactic rotation.
        extras["orientation"] = (0.0, float(np.radians(tilt)), 0.0)
    return extras


def _halo_params(p: Mapping[str, float], r_t: float) -> dict:
    return dict(
        type="Spheroid",
        scaleRadius=p["a_TwoPowerTriaxial_halo"],
        densityNorm=p["rho_TwoPowerTriaxial_halo"],
        gamma=p["gamma_TwoPowerTriaxial_halo"],
        beta=p["beta_TwoPowerTriaxial_halo"],
        cutoffStrength=2,
        outerCutoffRadius=r_t,
        axisRatioZ=p["q_TwoPowerTriaxial_halo"],
        **_halo_shape_extras(p),
    )


def _halo_params_m200c(agama, p: Mapping[str, float], cfg: Mapping) -> dict:
    """Halo ``Spheroid`` params from (virial mass, concentration) instead of (densityNorm, scale
    radius) — the McMillan (2017) parameterization.

    Sampled globals (both in log space, so the stock uniform/normal prior machinery and the
    analytic compositional prior score need no changes):
      * ``log10_M200_TwoPowerTriaxial_halo`` : log10(M200 / Msun); M200 is the mass enclosed at
        ``r200`` where the mean density is ``Delta_mass * rho_crit``.
      * ``ln_cvprime_TwoPowerTriaxial_halo`` : ln(c_v'); ``c_v' = r_v' / r_{-2}`` at the McMillan
        ``Delta_c ~ 94 rho_crit`` "virial" overdensity (prior ``ln c_v' ~ N(2.56, 0.272)``).

    Conversion (cosmology from ``cfg``; McMillan H0=70.4, Delta_mass=200, Delta_c=94):
        rho_crit = 3 H0^2 / (8 pi G)
        r200  = [3 M200 / (4 pi Delta_mass rho_crit)]^(1/3)
        c200  = convert_concentration(c_v', Delta_c, Delta_mass)
        r_h   = r200 / [c200 (2 - gamma)]      # r_{-2} = (2 - gamma) r_h; gamma capped < 2 by prior
        rho_0 = M200 / enclosedMass(r200)      # unit-norm solve: exact incl. cutoff + flattening q
    Validated to reproduce McMillan Table 3 (M200=1.3e12, c_v'=15.4, gamma=1 -> r_h=19.6 kpc,
    rho_0=8.5e6 Msun/kpc^3).
    """
    r_t = float(cfg["halo_r_t_kpc"])
    H0 = float(cfg["halo_H0_kms_mpc"]) / 1000.0  # km/s/Mpc -> km/s/kpc
    D_mass = float(cfg["halo_Delta_mass"])
    D_c = float(cfg["halo_Delta_c"])
    rho_crit = 3.0 * H0**2 / (8.0 * np.pi * agama.G)  # Msun/kpc^3

    M200 = 10.0 ** float(p["log10_M200_TwoPowerTriaxial_halo"])
    cvp = float(np.exp(float(p["ln_cvprime_TwoPowerTriaxial_halo"])))
    gamma = float(p["gamma_TwoPowerTriaxial_halo"])

    r200 = (3.0 * M200 / (4.0 * np.pi * D_mass * rho_crit)) ** (1.0 / 3.0)
    c200 = convert_concentration(cvp, D_c, D_mass)
    r_h = r200 / (c200 * (2.0 - gamma))

    shape = dict(
        type="Spheroid",
        scaleRadius=r_h,
        densityNorm=1.0,
        gamma=gamma,
        beta=float(p["beta_TwoPowerTriaxial_halo"]),
        cutoffStrength=2,
        outerCutoffRadius=r_t,
        axisRatioZ=float(p["q_TwoPowerTriaxial_halo"]),
        **_halo_shape_extras(p),
    )
    shape["densityNorm"] = M200 / agama.Potential(**shape).enclosedMass(r200)
    return shape


Z_SUN_KPC = 0.0208  # Solar height above the plane (astropy/agama default; not varied)


def _solar_frame(agama, pot_host, p: Mapping[str, float]) -> tuple[float, float, float, float, float]:
    """The row's solar reference frame ``(R0, v_sun_x, v_sun_y, v_sun_z, z_sun)``.

    Optional priors (each defaulting to its fixed literature value, so configs that do not declare
    them are unaffected):

    * ``R0_Sun``            — Solar galactocentric radius [kpc]; N(8.178, 0.026), GRAVITY (2019).
    * ``U_Sun``/``V_Sun``/``W_Sun`` — the Sun's PECULIAR velocity [km/s] w.r.t. the local circular
      orbit; N(11.1, 1.25) / N(12.24, 2.05) / N(7.25, 0.62), Schonrich, Binney & Dehnen (2010).

    ``V_Sun`` is peculiar, so the frame's azimuthal velocity is ``v_circ(R0) + V_Sun`` evaluated in
    **this row's own potential** — the Sun's motion is not independent of the mass model, and tying
    them makes the frame self-consistent rather than a free offset. (Sanity check: with V_circ ~ 233
    km/s this gives ~245 km/s, matching astropy's default galcen_v_sun of (12.9, 245.6, 7.78).)

    The same frame is used for the progenitor's observed-ICRS -> Galactocentric conversion, for the
    Galactocentric -> ICRS projection of the particles, and for the R0-dependent ancillary
    observables, so a varied Solar position cannot desynchronise them.
    """
    R0 = float(p.get("R0_Sun", R0_KPC))
    v_circ = float(vcirc_from_potential(pot_host, R0)[0])
    return (
        R0,
        float(p.get("U_Sun", 11.1)),
        v_circ + float(p.get("V_Sun", 12.24)),
        float(p.get("W_Sun", 7.25)),
        Z_SUN_KPC,
    )


def _host_potential(agama, p: Mapping[str, float], pot_cfg: Mapping | None = None):
    """Assemble the Milky Way host potential from one parameter row.

    Legacy model (``pot_cfg=None``): fixed bulge + two-power triaxial halo (untruncated) + one
    isothermal (sech^2) exponential-radial disk. Ibata model (``pot_cfg`` from the Ibata sim
    config): additionally truncates the halo at ``halo_r_t_kpc`` (1000 kpc), adds the fixed HI &
    H2 gas disks, adds a free thick stellar disk, and (optionally) switches the stellar disks to an
    exponential vertical profile — see ``new_constrains.md``.
    """
    cfg = _resolve_pot_cfg(pot_cfg)
    hsign = -1.0 if cfg["disk_vertical"] == "exponential" else 1.0
    # The bulge amplitude is normally a fixed config scalar. A config MAY instead declare a prior on
    # `rho_Bulge`, in which case it is read per row: the bulge-disc trade-off at R ~ 5-8 kpc feeds
    # straight into Sigma_Disk / r_Disk, and pinning all five bulge parameters forces the halo cusp
    # to absorb any inner-mass mismatch (one of the reasons gamma rails).
    bulge_norm = float(p.get("rho_Bulge", cfg["bulge_density_norm"]))
    bulge = {**BULGE_PARAMS, "densityNorm": bulge_norm}
    components = [bulge]
    if cfg["gas_disks"]:
        components += [GAS_HI_PARAMS, GAS_H2_PARAMS]
    if str(cfg["halo_parameterization"]) == "m200_c":
        components.append(_halo_params_m200c(agama, p, cfg))
    else:
        components.append(_halo_params(p, float(cfg["halo_r_t_kpc"])))
    components.append(_thin_disk_params(p, hsign))
    if cfg["thick_disk"]:
        components.append(_thick_disk_params(p, hsign))
    return agama.Potential(*components)


def _stellar_disk_potential(agama, p: Mapping[str, float], pot_cfg: Mapping | None = None):
    """The stellar-disk component(s) alone (thin [+ thick]) — the ``disk_pot`` for the vertical
    stellar-density observable rho(z), so gas/halo do not contaminate the stellar profile."""
    cfg = _resolve_pot_cfg(pot_cfg)
    hsign = -1.0 if cfg["disk_vertical"] == "exponential" else 1.0
    disks = [_thin_disk_params(p, hsign)]
    if cfg["thick_disk"]:
        disks.append(_thick_disk_params(p, hsign))
    return agama.Potential(*disks)



def _rj_vj_R(agama, pot_host, orbit_sat: np.ndarray, mass_sat: float):
    """Jacobi radius, velocity offset, and host->satellite rotation matrices along the orbit."""
    N = len(orbit_sat)
    x, y, z, vx, vy, vz = orbit_sat.T
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = (x * x + y * y + z * z) ** 0.5
    L = (Lx * Lx + Ly * Ly + Lz * Lz) ** 0.5
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = x / r
    R[:, 0, 1] = y / r
    R[:, 0, 2] = z / r
    R[:, 2, 0] = Lx / L
    R[:, 2, 1] = Ly / L
    R[:, 2, 2] = Lz / L
    R[:, 1, 0] = R[:, 0, 2] * R[:, 2, 1] - R[:, 0, 1] * R[:, 2, 2]
    R[:, 1, 1] = R[:, 0, 0] * R[:, 2, 2] - R[:, 0, 2] * R[:, 2, 0]
    R[:, 1, 2] = R[:, 0, 1] * R[:, 2, 0] - R[:, 0, 0] * R[:, 2, 1]
    der = pot_host.eval(orbit_sat[:, 0:3], der=True)
    d2Phi_dr2 = -(
        x**2 * der[:, 0] + y**2 * der[:, 1] + z**2 * der[:, 2]
        + 2 * x * y * der[:, 3] + 2 * y * z * der[:, 4] + 2 * z * x * der[:, 5]
    ) / r**2
    Omega = L / r**2
    # ``mass_sat`` may be a scalar (mass held fixed) or a per-seed array (mass loss): the Jacobi
    # radius then shrinks along the orbit as the progenitor dissolves.
    rj = (agama.G * np.asarray(mass_sat) / (Omega**2 - d2Phi_dr2)) ** (1.0 / 3)
    vj = Omega * rj
    return rj, vj, R


def _ic_particle_spray(
    orbit_sat: np.ndarray, rj: np.ndarray, vj: np.ndarray, R: np.ndarray,
    rng: np.random.Generator, gala_modified: bool = True,
) -> np.ndarray:
    """Fardal+2015 initial conditions for particles escaping through the Lagrange points."""
    N = len(rj)
    rj = np.repeat(rj, 2) * np.tile([1, -1], N)
    vj = np.repeat(vj, 2) * np.tile([1, -1], N)
    R = np.repeat(R, 2, axis=0)
    mean_x = 2.0
    disp_x = 0.5 if gala_modified else 0.4
    disp_z = 0.5
    mean_vy = 0.3
    disp_vy = 0.5 if gala_modified else 0.4
    disp_vz = 0.5
    rx = rng.normal(size=2 * N) * disp_x + mean_x
    rz = rng.normal(size=2 * N) * disp_z * rj
    rvy = (rng.normal(size=2 * N) * disp_vy + mean_vy) * vj * (rx if gala_modified else 1)
    rvz = rng.normal(size=2 * N) * disp_vz * vj
    rx *= rj
    offset_pos = np.column_stack([rx, rx * 0, rz])
    offset_vel = np.column_stack([rx * 0, rvy, rvz])
    ic_stream = np.tile(orbit_sat, 2).reshape(2 * N, 6)
    ic_stream[:, 0:3] += np.einsum("ni,nij->nj", offset_pos, R)
    ic_stream[:, 3:6] += np.einsum("ni,nij->nj", offset_vel, R)
    return ic_stream


# Chen, Gnedin & Li (2024) release model (arXiv:2408.01496), ported 1:1 from gala's
# ``ChenStreamDF._sample`` (gala/dynamics/mockstream/df.pyx). The 6D draw is
# (r, phi, theta, v, alpha, beta): a radial scale r*rj, sky angles (phi, theta) for the
# escape *position*, a velocity scale v (fixed to the local escape speed), and angles
# (alpha, beta) for the escape *velocity* — all in the satellite frame whose basis
# (radial, L x radial, L) matches ``_rj_vj_R`` here, so the same ``einsum(..., R)`` maps
# back to Galactocentric coordinates as for the Fardal path.
_CHEN_MEAN = np.array([1.6, -30.0, 0.0, 1.0, 20.0, 0.0])  # r, phi[deg], theta[deg], v, alpha[deg], beta[deg]
_CHEN_COV = np.diag([0.1225, 529.0, 144.0, 0.0, 400.0, 484.0])
_CHEN_COV[0, 4] = _CHEN_COV[4, 0] = -4.9  # covariance between r and alpha


def _ic_chen_spray(
    orbit_sat: np.ndarray, rj: np.ndarray, R: np.ndarray, rng: np.random.Generator,
    mass_sat: float, G: float,
) -> np.ndarray:
    """Chen+2024 initial conditions: one trailing + one leading particle per orbit seed.

    Mirrors gala's ``ChenStreamDF``: the leading tail is the trailing draw with 180 deg added
    to the position angle ``phi`` and the velocity angle ``alpha``. Particles are interleaved
    (``2k`` trailing, ``2k+1`` leading) so they line up with ``np.repeat(time_sat, 2)`` in
    :func:`_spray_stream`, exactly like the Fardal path.
    """
    N = len(rj)
    trail = rng.multivariate_normal(_CHEN_MEAN, _CHEN_COV, size=N, check_valid="ignore")
    lead = rng.multivariate_normal(_CHEN_MEAN, _CHEN_COV, size=N, check_valid="ignore")
    lead[:, 1] += 180.0  # phi   -> leading tail
    lead[:, 4] += 180.0  # alpha -> leading tail
    s = np.empty((2 * N, 6))
    s[0::2] = trail
    s[1::2] = lead

    rj2 = np.repeat(rj, 2)
    Dr = s[:, 0] * rj2
    # scalar mass (fixed progenitor) or one mass per orbit seed (mass loss); the interleaved
    # trailing/leading pair of a seed shares its mass, hence the repeat.
    m2 = np.repeat(np.asarray(mass_sat), 2) if np.ndim(mass_sat) else mass_sat
    with np.errstate(invalid="ignore"):
        Dv = s[:, 3] * np.sqrt(2.0 * G * m2 / Dr)  # local escape speed at Dr
    phi, theta = np.deg2rad(s[:, 1]), np.deg2rad(s[:, 2])
    alpha, beta = np.deg2rad(s[:, 4]), np.deg2rad(s[:, 5])
    offset_pos = Dr[:, None] * np.column_stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)]
    )
    offset_vel = Dv[:, None] * np.column_stack(
        [np.cos(beta) * np.cos(alpha), np.cos(beta) * np.sin(alpha), np.sin(beta)]
    )
    R2 = np.repeat(R, 2, axis=0)
    ic_stream = np.repeat(orbit_sat, 2, axis=0)
    ic_stream[:, 0:3] += np.einsum("ni,nij->nj", offset_pos, R2)
    ic_stream[:, 3:6] += np.einsum("ni,nij->nj", offset_vel, R2)
    return ic_stream


def _mass_track(
    mass_initial: float, mass_final: float, time_sat: np.ndarray, time_total: float, law: str
) -> np.ndarray:
    """Progenitor bound mass at each release time, from ``mass_initial`` at ``t = -time_total`` to
    ``mass_final`` at the present day.

    ``time_sat`` is ascending and negative (agama time units), so ``u = 1 + t/time_total`` is the
    fraction of the stripping window elapsed: 0 at the first release, 1 today.

    * ``linear``      constant mass-loss rate — the first-order behaviour of a cluster on the
      dissolution track (Baumgardt & Makino 2003), and the honest interpolation between the two
      endpoints we actually have numbers for (an initial-mass estimate and a measured present-day
      mass).
    * ``exponential`` constant *fractional* loss rate; steeper early, gentler late.

    ``mass_final >= mass_initial`` degenerates to the fixed-mass case, which is correct: the prior
    lower bound on the initial mass is the observed present-day mass.
    """
    u = np.clip(1.0 + np.asarray(time_sat, dtype=float) / float(time_total), 0.0, 1.0)
    m_i, m_f = float(mass_initial), float(mass_final)
    if m_f >= m_i:
        return np.full(u.shape, m_i)
    if law == "exponential":
        return m_i * (m_f / m_i) ** u
    if law != "linear":
        raise ValueError(f"unknown mass-loss law {law!r} (expected 'linear' or 'exponential')")
    return m_i + (m_f - m_i) * u


def _spray_stream(
    agama, pot_host, posvel_sat: np.ndarray, mass_sat: float, radius_sat: float,
    time_total: float, num_particles: int, rng: np.random.Generator,
    method: str = "fardal", mass_final: float | None = None, mass_loss: str = "linear",
) -> np.ndarray:
    """Particle-spray stream including the progenitor's own (moving Plummer) potential.

    With ``mass_final`` given, the progenitor loses mass over the integration per ``mass_loss``
    (see :func:`_mass_track`): the Jacobi radius, the Chen+2024 escape speed and the moving Plummer
    potential all follow the same track, the last via agama's time-dependent ``scale`` modifier.
    ``mass_final=None`` holds ``mass_sat`` fixed for the whole integration (the original behaviour).

    Invalid seeds (undefined Jacobi radius where ``Omega^2 - d2Phi/dr2 < 0``) stay NaN so the
    output always has shape ``(num_particles, 6)``; downstream NaN cleaning drops those rows.
    """
    N = num_particles // 2
    time_sat, orbit_sat = agama.orbit(
        potential=pot_host, ic=posvel_sat, time=-time_total, trajsize=N + 1, accuracy=1e-10
    )
    time_sat = time_sat[1:][::-1]
    orbit_sat = orbit_sat[1:][::-1]

    mass_track = (
        None if mass_final is None
        else _mass_track(mass_sat, mass_final, time_sat, time_total, mass_loss)
    )
    mass_arg = mass_sat if mass_track is None else mass_track
    rj, vj, R = _rj_vj_R(agama, pot_host, orbit_sat, mass_arg)
    if method == "chen":
        ic_stream = _ic_chen_spray(orbit_sat, rj, R, rng, mass_arg, agama.G)
    else:
        ic_stream = _ic_particle_spray(orbit_sat, rj, vj, R, rng)
    time_seed = np.repeat(time_sat, 2)

    sat_kwargs = {}
    if mass_track is not None:  # (t, mass factor, size factor); size held fixed
        sat_kwargs["scale"] = np.column_stack(
            [time_sat, mass_track / float(mass_sat), np.ones_like(mass_track)]
        )
    pot_sat = agama.Potential(
        type="Plummer",
        mass=mass_sat,
        scaleRadius=radius_sat,
        center=np.column_stack([time_sat, orbit_sat]),
        **sat_kwargs,
    )
    pot_total = agama.Potential(pot_host, pot_sat)

    valid = np.isfinite(ic_stream).all(axis=1)
    xv = np.full((len(ic_stream), 6), np.nan)
    if valid.any():
        result = agama.orbit(
            potential=pot_total,
            ic=ic_stream[valid],
            timestart=time_seed[valid],
            time=-time_seed[valid],
            trajsize=1,
            accuracy=1e-10,
        )
        xv[valid] = np.vstack(result[:, 1])
    return xv



def window_subsample(
    projected: np.ndarray,
    j: np.ndarray,
    windows: Mapping[int, Mapping[str, float]],
    max_particles: int,
    pad_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep only the stars inside each row's stream observation window, at most ``max_particles``
    per row (a uniform random subset when there are more), and pad the rest with ``pad_value``.

    ``projected`` is ``(n, P, 6)`` ICRS (ra, dec, ...); ``j`` the ``(n,)`` stream indices;
    ``windows`` maps a stream index to ``{ra_min, ra_max, dec_min, dec_max}`` — the SAME table the
    ``observational_window`` augmentation applies at training time, so the stored subset is exactly
    what that step would have attended to, and the padding (a finite sentinel outside every window)
    is what it masks out. Rows with any non-finite star (a failed simulation) are returned as a full
    NaN row so ``drop_nan`` still removes them. Output ``(n, max_particles, 6)`` float32.
    """
    n = projected.shape[0]
    out = np.full((n, int(max_particles), 6), float(pad_value), dtype=np.float32)
    for i in range(n):
        row = projected[i]
        if not np.isfinite(row).all():
            out[i] = np.nan
            continue
        w = windows[int(j[i])]
        ra, dec = row[:, 0], row[:, 1]
        keep = np.flatnonzero(
            (ra >= w["ra_min"]) & (ra <= w["ra_max"]) & (dec >= w["dec_min"]) & (dec <= w["dec_max"])
        )
        if len(keep) > max_particles:
            keep = np.sort(rng.choice(keep, size=int(max_particles), replace=False))
        out[i, : len(keep)] = row[keep]
    return out

def _vcirc(pot_host, obs_r: np.ndarray) -> np.ndarray:
    """Model circular velocity [km/s] at the observed radii; NaN where v^2 < 0."""
    points = np.column_stack((obs_r, np.zeros_like(obs_r), np.zeros_like(obs_r)))
    v2 = -obs_r * pot_host.force(points)[:, 0]
    return np.sqrt(np.where(v2 > 0, v2, np.nan))


@quiet_worker
def _vcirc_accept_worker(
    rows: list, obs_r: np.ndarray, bands: list, pot_cfg: Mapping | None = None
) -> np.ndarray:
    """joblib worker: does each parameter row's model rotation curve pass the rejection cut?

    A row is accepted iff it passes *every* band. Each band selects a set of radial bins
    (``bin_mask``) and requires ``stat`` (median or max) of a per-bin deviation to fall below
    ``threshold``: ``|vc - vc_ref| / vc_ref`` for a ``fracdev`` band, ``|vc - vc_ref| / sigma``
    for a ``sigma`` band. Rows whose potential fails to build or yields a NaN circular velocity
    in any used bin are rejected.
    """
    agama = _agama()
    agama.setNumThreads(1)
    out = np.zeros(len(rows), dtype=bool)
    for i, p in enumerate(rows):
        try:
            vc = _vcirc(_host_potential(agama, p, pot_cfg), obs_r)
        except Exception:
            continue
        ok = True
        for b in bands:
            v = vc[b["bin_mask"]]
            if np.isnan(v).any():
                ok = False
                break
            if b["criterion"] == "sigma":
                dev = np.abs(v - b["vc_ref"]) / b["sigma"]
            else:
                dev = np.abs(v - b["vc_ref"]) / b["vc_ref"]
            reduce = np.median if b["stat"] == "median" else np.max
            if not bool(reduce(dev) < b["threshold"]):
                ok = False
                break
        out[i] = ok
    return out


def _ancillary_observables(
    agama, pot_host, p: Dict[str, float], pot_cfg: Mapping | None, spec: Mapping | None,
    r0: float | None = None,
) -> Dict[str, np.ndarray]:
    """Ibata (2023) potential-derived observables for one row, per ``spec`` (an empty/None spec
    computes nothing). ``spec`` carries the requested observable names + their fixed grids:
    ``{"names": [...], "l_deg": array, "z_kpc": array}``. Deterministic force/density evaluations
    on the assembled potential — no stream simulation.

    All three observables are anchored at the Solar radius, so ``r0`` (this row's ``R0_Sun``) is
    threaded through: the terminal-velocity tangent points are at ``R0 |sin l|``, and Sigma(1.1 kpc)
    and rho(z) are evaluated at R0. Defaults to the fixed ``R0_KPC``."""
    if not spec or not spec.get("names"):
        return {}
    R0 = R0_KPC if r0 is None else float(r0)
    names = set(spec["names"])
    out: Dict[str, np.ndarray] = {}
    if "vterm" in names:
        out["vterm_kms"] = terminal_velocity(pot_host, spec["l_deg"], R0=R0).astype(float)
    if "sigma_z" in names:
        out["sigma_z"] = np.array([surface_density(pot_host, R=R0)], dtype=float)
    if "rho_z" in names:
        disk_pot = _stellar_disk_potential(agama, p, pot_cfg)
        out["rho_z"] = vertical_density_profile(disk_pot, spec["z_kpc"], R=R0).astype(float)
    return out


@quiet_worker
def _simulate_one(
    p: Dict[str, float], n_particles: int, obs_r: np.ndarray, seed: int,
    spray_method: str = "fardal", pot_cfg: Mapping | None = None,
    ancillary: Mapping | None = None, mass_loss: str | None = None,
):
    """joblib worker: one stream realization + the rotation curve of its potential (+ the optional
    Ibata ancillary observables of that potential).

    ``mass_loss`` (``linear``/``exponential``) makes the progenitor shed mass from its initial
    ``m_progenitor`` down to the row's ``m_progenitor_final`` over the integration; ``None`` holds
    the mass fixed. The trailing ``None`` of the returned tuple is the present-day bound progenitor
    mass, which only the restricted-N-body worker can report — spray *imposes* its final mass, so
    reporting it back would dress an input up as a measurement."""
    agama = _agama()
    rng = np.random.default_rng(seed)
    # getUnits()['time'] is a float [Myr] normally, but an astropy Quantity once astropy has
    # been imported in this process (agama >= 1.0.157) — take .value in that case.
    tu = agama.getUnits()["time"]
    time_unit_gyr = float(getattr(tu, "value", tu)) / 1e3

    pot_host = _host_potential(agama, p, pot_cfg)
    frame = _solar_frame(agama, pot_host, p)

    l0, b0, pml0, pmb0 = agama.transformCelestialCoords(
        agama.fromICRStoGalactic,
        p["ra"] * np.pi / 180,
        p["dec"] * np.pi / 180,
        p["mu_ra_cosdec"],
        p["mu_dec"],
    )
    posvel_sat = np.array(
        agama.getGalactocentricFromGalactic(
            l0, b0, p["r"], pml0 * 4.74, pmb0 * 4.74, p["vr"],
            galcen_distance=frame[0], galcen_v_sun=frame[1:4], z_sun=frame[4],
        )
    )

    xv = _spray_stream(
        agama,
        pot_host,
        posvel_sat,
        mass_sat=p["m_progenitor"],
        radius_sat=p["a_progenitor"] / 1e3,  # pc -> kpc
        time_total=p["t_end"] / time_unit_gyr,
        num_particles=n_particles,
        rng=rng,
        method=spray_method,
        mass_final=None if mass_loss is None else float(p["m_progenitor_final"]),
        mass_loss=mass_loss or "linear",
    )
    anc = _ancillary_observables(agama, pot_host, p, pot_cfg, ancillary, r0=frame[0])
    # When the halo is parameterized by (M200, c_v'), also return the (densityNorm, scaleRadius)
    # actually handed to AGAMA for this row, so they can be stored in the dataset for traceability
    # (they are NOT inferred — the identity-prior rho/a stay fixed constants). None otherwise.
    halo_derived = None
    if str(_resolve_pot_cfg(pot_cfg)["halo_parameterization"]) == "m200_c":
        h = _halo_params_m200c(agama, p, pot_cfg)
        halo_derived = (float(h["densityNorm"]), float(h["scaleRadius"]))
    return xv, _vcirc(pot_host, obs_r), anc, halo_derived, None, frame


@register_simulator("stream_agama")
class AgamaStreamSimulator(BaseSimulator):
    """Stellar streams in a parametrized Milky Way potential, simulated with AGAMA."""

    @property
    def _priors_global(self) -> Dict[str, dict]:
        return self._as_dict(self.params["priors_global"])

    @property
    def _priors_local(self) -> Dict[str, dict]:
        return self._as_dict(self.params["priors_local"])

    @property
    def target_streams(self) -> Dict[str, int]:
        return {str(k): int(v) for k, v in self._as_dict(self.params["target_streams"]).items()}

    @property
    def _n_particles(self) -> int:
        return int(self.params.get("n_particles", 1000))

    @property
    def _store_window_subsample(self) -> dict | None:
        """``params.store_window_subsample`` -> ``{max_particles, pad_value, windows}`` with the
        window table keyed by stream index, or ``None`` (store the full cloud, the default)."""
        cfg = self.params.get("store_window_subsample")
        if cfg is None:
            return None
        cfg = self._as_dict(cfg)
        table = self._as_dict(cfg["observational_window"])
        names = self.target_streams
        missing = set(names) - set(map(str, table))
        if missing:
            raise KeyError(f"store_window_subsample.observational_window lacks streams {missing}")
        windows = {
            idx: {k: float(self._as_dict(table[name])[k]) for k in ("ra_min", "ra_max", "dec_min", "dec_max")}
            for name, idx in names.items()
        }
        pad = float(cfg.get("pad_value", -999.0))
        for w in windows.values():
            if w["ra_min"] <= pad <= w["ra_max"] or w["dec_min"] <= pad <= w["dec_max"]:
                raise ValueError("store_window_subsample.pad_value must lie outside every window")
        return {"max_particles": int(cfg["max_particles"]), "pad_value": pad, "windows": windows}

    @property
    def _n_workers(self) -> int:
        return int(self.params.get("n_workers", 8))

    @property
    def _obs_r_split(self) -> float:
        """Split radius for the extended (Zhou u Huang) rotation-curve grid."""
        return float(self.params.get("obs_r_split_kpc", float(OBS_R_KPC.max())))

    def _custom_rotation_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``obs_r_grid: custom`` -> the (radii, observed v_c, 1-sigma) table given VERBATIM in the
        config as ``obs_r_kpc`` / ``obs_vc_kms`` / ``obs_sigma_vc``. All three are required and
        must align, so a table typo cannot silently produce a mis-registered observable."""
        try:
            r = np.asarray(self.params["obs_r_kpc"], dtype=float)
            vc = np.asarray(self.params["obs_vc_kms"], dtype=float)
            sig = np.asarray(self.params["obs_sigma_vc"], dtype=float)
        except KeyError as exc:
            raise KeyError(
                "obs_r_grid=custom needs obs_r_kpc, obs_vc_kms and obs_sigma_vc lists in "
                "simulator.params"
            ) from exc
        if not (r.ndim == 1 and r.shape == vc.shape == sig.shape and len(r) > 0):
            raise ValueError(
                f"obs_r_grid=custom: obs_r_kpc/obs_vc_kms/obs_sigma_vc must be equal-length 1-D "
                f"lists, got {r.shape}/{vc.shape}/{sig.shape}"
            )
        if np.any(np.diff(r) <= 0):
            raise ValueError("obs_r_grid=custom: obs_r_kpc must be strictly increasing")
        return r, vc, sig

    @property
    def obs_r_kpc(self) -> np.ndarray:
        """Radii the model rotation curve is evaluated on (also the ``vcirc_kms`` grid).

        ``obs_r_grid: extended`` -> Zhou below ``obs_r_split_kpc`` u Huang beyond (50 radii,
        the single source of truth being ``stream_common.extended_rotation_curve``);
        ``obs_r_grid: custom`` -> the explicit ``obs_r_kpc`` table (with ``obs_vc_kms`` and
        ``obs_sigma_vc``, all required); otherwise an explicit ``obs_r_kpc`` list overrides
        the Zhou grid.
        """
        grid = str(self.params.get("obs_r_grid", "") or "")
        if grid == "extended":
            return extended_rotation_curve(self._obs_r_split)[0]
        if grid == "custom":
            return self._custom_rotation_curve()[0]
        return np.asarray(self.params.get("obs_r_kpc", OBS_R_KPC), dtype=float)

    @property
    def obs_sigma_vc(self) -> np.ndarray:
        """Per-bin observed 1-sigma on the rotation curve, aligned with ``obs_r_kpc``.

        ``obs_r_grid: extended`` -> the Zhou u Huang errors (Zhou below ``obs_r_split_kpc``,
        Huang beyond), so the training-time ``add_noise_to_vcirc`` gives the outer (Huang) radii
        their own heteroscedastic errors rather than Zhou's ~0.2 km/s inner errors; an explicit
        ``obs_sigma_vc`` list overrides; otherwise the Zhou grid. Mirrors :meth:`obs_r_kpc` so the
        two stay aligned and the simulator remains the single source of truth for the vcirc grid.
        """
        grid = str(self.params.get("obs_r_grid", "") or "")
        if grid == "extended":
            return extended_rotation_curve(self._obs_r_split)[2]
        if grid == "custom":
            return self._custom_rotation_curve()[2]
        return np.asarray(self.params.get("obs_sigma_vc", OBS_SIGMA_VC), dtype=float)

    @property
    def obs_vc_kms(self) -> np.ndarray:
        """Observed Milky Way circular velocity aligned with ``obs_r_kpc`` — the fixed curve
        ``attach_observed_vcirc`` supplies to real-data evaluation (simulated data carry their own
        model curve). ``obs_r_grid: extended`` -> Zhou below the split u Huang beyond; otherwise
        the Zhou grid. Mirrors :meth:`obs_r_kpc` / :meth:`obs_sigma_vc`."""
        grid = str(self.params.get("obs_r_grid", "") or "")
        if grid == "extended":
            return extended_rotation_curve(self._obs_r_split)[1]
        if grid == "custom":
            return self._custom_rotation_curve()[1]
        return np.asarray(self.params.get("obs_vc_kms", OBS_VC_KMS), dtype=float)

    # ------------------------------------------------------------------------------------- #
    # Potential model + Ibata ancillary observables (config-driven; legacy defaults are no-ops)
    # ------------------------------------------------------------------------------------- #

    @property
    def _pot_cfg(self) -> dict:
        """Host-potential configuration threaded to the joblib workers. Legacy default (all keys
        absent) reproduces the original bulge + untruncated halo + one isothermal disk; the Ibata
        sim config sets ``halo_r_t_kpc`` / ``gas_disks`` / ``thick_disk`` / ``disk_vertical`` /
        ``bulge_density_norm``."""
        return dict(
            halo_r_t_kpc=float(self.params.get("halo_r_t_kpc", float("inf"))),
            gas_disks=bool(self.params.get("gas_disks", False)),
            thick_disk=bool(self.params.get("thick_disk", False)),
            disk_vertical=str(self.params.get("disk_vertical", "isothermal")),
            bulge_density_norm=float(
                self.params.get("bulge_density_norm", BULGE_PARAMS["densityNorm"])
            ),
            halo_parameterization=str(self.params.get("halo_parameterization", "rho_a")),
            halo_H0_kms_mpc=float(self.params.get("halo_H0_kms_mpc", 70.4)),
            halo_Delta_mass=float(self.params.get("halo_Delta_mass", 200.0)),
            halo_Delta_c=float(self.params.get("halo_Delta_c", 94.0)),
        )

    @property
    def _ancillary_names(self) -> list[str]:
        """Requested Ibata ancillary observables (``params.ancillary_observables``); empty by
        default so existing stream configs compute nothing new (zero overhead)."""
        node = self.params.get("ancillary_observables", None)
        if not node:
            return []
        names = list(self._as_dict(node)) if not isinstance(node, (list, tuple)) else list(node)
        valid = {"vterm", "sigma_z", "rho_z"}
        bad = set(names) - valid
        if bad:
            raise ValueError(f"Unknown ancillary_observables {sorted(bad)}; valid: {sorted(valid)}")
        return names

    @property
    def vterm_l_deg(self) -> np.ndarray:
        """Galactic longitudes [deg] the terminal-velocity curve is evaluated on (single source of
        truth, not stored in the npz — like ``obs_r_kpc`` for vcirc)."""
        node = self.params.get("vterm_l_deg", None)
        return np.asarray(node, dtype=float) if node else np.asarray(VTERM_L_DEG, dtype=float)

    @property
    def rho_z_kpc(self) -> np.ndarray:
        """Heights z [kpc] the vertical stellar-density profile is evaluated on."""
        node = self.params.get("rho_z_kpc", None)
        return np.asarray(node, dtype=float) if node else np.asarray(RHO_Z_KPC, dtype=float)

    @property
    def _ancillary_spec(self) -> dict | None:
        """Spec passed to the joblib worker: requested names + their fixed grids (or None)."""
        names = self._ancillary_names
        if not names:
            return None
        return {"names": names, "l_deg": self.vterm_l_deg, "z_kpc": self.rho_z_kpc}

    @property
    def ancillary_observable_keys(self) -> list[str]:
        """Dataset keys the requested ancillary observables are stored under (``vterm``->``vterm_kms``,
        ``sigma_z``->``sigma_z``, ``rho_z``->``rho_z``). Empty when none requested. Used by the
        adapter/model presets and PPC to discover the extra modalities without hardcoding names."""
        key_of = {"vterm": "vterm_kms", "sigma_z": "sigma_z", "rho_z": "rho_z"}
        return [key_of[n] for n in self._ancillary_names]

    @staticmethod
    def _as_dict(node) -> dict:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node)

    # ------------------------------------------------------------------------------------- #
    # Declarations driving the adapter / compositional workflow
    # ------------------------------------------------------------------------------------- #

    @property
    def parameter_names(self) -> list[str]:
        return self.global_parameter_names + self.local_parameter_names

    @property
    def _marginalized(self) -> set[str]:
        """Names listed in ``params.marginalize``: parameters that are DRAWN from their prior and fed
        to the forward model, but excluded from the inferred set.

        This is the right treatment for genuine nuisances whose values we do not want the network to
        report — the Solar phase-space parameters (``R0_Sun``, ``U_Sun``, ``V_Sun``, ``W_Sun``) are
        the motivating case: varying them is what makes the posterior honest about the frame
        systematic, but they are external measurements, not something these three streams constrain.
        Because ``prior_spec_global`` slices by ``global_parameter_names``, a marginalized parameter
        also drops out of the compositional prior score, which is correct — no score term exists for
        a dimension that is not being inferred.
        """
        return {str(k) for k in (self.params.get("marginalize") or [])}

    @property
    def global_parameter_names(self) -> list[str]:
        marginalized = self._marginalized
        return [k for k in inferred_names(self._priors_global) if k not in marginalized]

    @property
    def local_parameter_names(self) -> list[str]:
        first_stream = next(iter(self._priors_local.values()))
        return inferred_names(first_stream)

    @property
    def context_keys(self) -> list[str]:
        return ["j"]

    @property
    def observable_keys(self) -> list[str]:
        return ["sim_data_projected", "vcirc_kms"]

    @property
    def prior_spec_global(self) -> Dict[str, dict]:
        """Prior spec of the inferred global parameters (used for compositional prior scores)."""
        return {k: self._priors_global[k] for k in self.global_parameter_names}

    @property
    def prior_spec_local(self) -> Dict[str, Dict[str, dict]]:
        """Per-stream prior spec of the inferred local parameters (drives their normalization)."""
        return {
            name: {k: spec[k] for k in self.local_parameter_names}
            for name, spec in self._priors_local.items()
        }

    # ------------------------------------------------------------------------------------- #
    # Sampling
    # ------------------------------------------------------------------------------------- #

    @property
    def _vcirc_rejection(self) -> dict | None:
        """Optional rotation-curve rejection prior (``params.vcirc_rejection``), else None.

        Keys: ``max_frac_dev`` (required), ``stat`` (median|max over radial bins, default
        median), ``r_min_kpc`` (default 5.5, matching the ``mask_vcirc_radii`` preprocessing).
        The cut truncates the prior with a hard indicator, which leaves the compositional
        prior score (``prior_score_from_spec``) valid unchanged inside the accepted region.
        """
        node = self.params.get("vcirc_rejection", None)
        return self._as_dict(node) if node else None

    def _build_accept_bands(self, cfg: Mapping, obs_r: np.ndarray) -> list:
        """Resolve ``vcirc_rejection`` into a list of band specs for the accept worker.

        ``bands`` mode (a list of ``{r_min_kpc, r_max_kpc, criterion, stat, ...}`` entries)
        sources the per-bin reference and 1-sigma from ``extended_rotation_curve`` (Zhou below
        the split, Huang above), which requires ``obs_r_grid: extended`` so the reference aligns
        with ``obs_r_kpc``. Without ``bands`` the legacy single fractional cut is reproduced.
        """
        if "bands" in cfg:
            r_ref, vc_ref_full, sig_full = extended_rotation_curve(self._obs_r_split)
            if not (len(r_ref) == len(obs_r) and np.allclose(r_ref, obs_r)):
                raise ValueError(
                    "banded vcirc_rejection requires obs_r_grid=extended so the Zhou/Huang "
                    "reference grid aligns with obs_r_kpc."
                )
            bands = []
            for b in cfg["bands"]:
                mask = (obs_r > float(b.get("r_min_kpc", 0.0))) & (
                    obs_r <= float(b.get("r_max_kpc", np.inf))
                )
                criterion = str(b.get("criterion", "fracdev"))
                threshold = float(b["max_n_sigma"] if criterion == "sigma" else b["max_frac_dev"])
                bands.append({
                    "bin_mask": mask,
                    "vc_ref": vc_ref_full[mask],
                    "sigma": sig_full[mask],
                    "criterion": criterion,
                    "threshold": threshold,
                    "stat": str(b.get("stat", "median")),
                })
            return bands

        use_bin = obs_r > float(cfg.get("r_min_kpc", 5.5))
        return [{
            "bin_mask": use_bin,
            "vc_ref": np.asarray(cfg.get("vc_ref_kms", OBS_VC_KMS), dtype=float)[use_bin],
            "sigma": None,
            "criterion": "fracdev",
            "threshold": float(cfg["max_frac_dev"]),
            "stat": str(cfg.get("stat", "median")),
        }]

    def _vcirc_accept_mask(self, draws: Mapping[str, np.ndarray]) -> np.ndarray:
        """Screen each draw's global potential against the observed rotation curve."""
        from joblib import Parallel, delayed

        obs_r = self.obs_r_kpc
        bands = self._build_accept_bands(self._vcirc_rejection, obs_r)

        n = len(np.asarray(next(iter(draws.values()))))
        gkeys = list(self._priors_global)
        rows = [
            {k: float(np.asarray(draws[k]).reshape(n, -1)[i, 0]) for k in gkeys}
            for i in range(n)
        ]
        n_jobs = min(self._n_workers, len(rows))
        chunks = [rows[i::n_jobs] for i in range(n_jobs)]
        pot_cfg = self._pot_cfg
        res = Parallel(n_jobs=n_jobs)(
            delayed(_vcirc_accept_worker)(c, obs_r, bands, pot_cfg) for c in chunks
        )
        mask = np.zeros(n, dtype=bool)
        for i, r in enumerate(res):
            mask[i::n_jobs] = r
        return mask

    def _rejection_sample(self, n: int, rng: np.random.Generator, sample_fn) -> Dict[str, np.ndarray]:
        """Draw with ``sample_fn`` until ``n`` rows pass the rotation-curve cut."""
        parts: list[Dict[str, np.ndarray]] = []
        accepted = drawn = 0
        while accepted < n:
            # Size the next batch from the measured acceptance rate (pessimistic start).
            rate = max(accepted / drawn if drawn else 0.1, 0.005)
            k = min(int(np.ceil((n - accepted) / rate * 1.2)), 100_000)
            draws = sample_fn(k, rng)
            mask = self._vcirc_accept_mask(draws)
            drawn += k
            accepted += int(mask.sum())
            if mask.any():
                parts.append({key: arr[mask] for key, arr in draws.items()})
            if drawn >= 5_000_000 and accepted < n:
                raise RuntimeError(
                    f"vcirc_rejection accepted only {accepted}/{n} rows after {drawn} prior "
                    "draws — the cut is too tight for this prior."
                )
        out = {key: np.concatenate([p[key] for p in parts], axis=0)[:n] for key in parts[0]}
        return out

    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        def draw(k: int, r: np.random.Generator) -> Dict[str, np.ndarray]:
            return sample_stream_prior(
                self._priors_global, self._priors_local, self.target_streams, k, r
            )

        if self._vcirc_rejection is None:
            return draw(n, rng)
        return self._rejection_sample(n, rng, draw)

    def _row_jobs(self, rows, seeds):
        """One joblib ``delayed`` call per row. The forward-model seam: subclasses swap the
        worker here (``stream_agama_rnbody``) while reusing ``simulate``'s dispatch + output
        assembly. Every worker must return the same 5-tuple
        ``(xv, vcirc, ancillary_dict, halo_derived_or_None, m_bound_or_None)``."""
        from joblib import delayed

        spray_method = str(self.params.get("spray_method", "fardal"))
        pot_cfg = self._pot_cfg
        ancillary = self._ancillary_spec
        # Absent (or null) `mass_loss` keeps the progenitor mass fixed, as every existing config
        # expects; a law name additionally requires an `m_progenitor_final` prior entry per stream.
        mass_loss = self.params.get("mass_loss")
        mass_loss = None if mass_loss is None else str(mass_loss)
        return [
            delayed(_simulate_one)(
                row, self._n_particles, self.obs_r_kpc, int(seed), spray_method, pot_cfg,
                ancillary, mass_loss,
            )
            for row, seed in zip(rows, seeds)
        ]

    def simulate(
        self, params: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        from joblib import Parallel

        n = len(np.asarray(next(iter(params.values()))))
        rows = [
            {k: float(np.asarray(v).reshape(n, -1)[i, 0]) for k, v in params.items()}
            for i in range(n)
        ]
        seeds = rng.integers(0, 2**31 - 1, size=n)

        # batch_size=1: forward-model rows have highly variable cost (restricted N-body rows,
        # especially the long t_end streams, run far longer than spray rows). joblib's default
        # batch_size='auto' grows the batch after fast early rows and then dispatches a few large
        # batches to a handful of workers, starving the rest and serializing the slow tail. One
        # row per dispatch keeps all n_workers busy; the per-task overhead is negligible next to a
        # multi-second row.
        with joblib_row_progress():  # advance the run_chunked bar once per finished row
            results = Parallel(n_jobs=self._n_workers, batch_size=1)(
                self._row_jobs(rows, seeds)
            )
        xv = np.stack([r[0] for r in results], axis=0)  # (n, n_particles, 6)
        vcirc = np.stack([r[1] for r in results], axis=0)[..., None]  # (n, n_radii, 1)

        # Project through each row's OWN solar frame when the Solar phase-space parameters are being
        # varied; otherwise keep astropy's default frame so existing configs are byte-for-byte
        # unchanged. Reusing the worker's frame (rather than rebuilding it) guarantees the projection
        # and the progenitor's ICRS -> Galactocentric conversion cannot desynchronise.
        frames = np.asarray([r[5] for r in results], dtype=float)  # (n, 5)
        varies_solar = any(k in params for k in ("R0_Sun", "U_Sun", "V_Sun", "W_Sun"))

        projected = sky_projection(xv, frames if varies_solar else None)
        sub = self._store_window_subsample
        if sub is None:
            out = {"sim_data_carthesian": xv, "sim_data_projected": projected, "vcirc_kms": vcirc}
        else:
            # Store only the in-window stars (capped, float32, sentinel-padded) and drop the
            # Cartesian copy: at 1e4 particles the full cloud is ~720 KB/row while the training
            # observation model keeps ~200 in-window stars — see `store_window_subsample` in the
            # config for the rationale and the window table (== the augmentation's).
            j = np.asarray(params["j"], dtype=float).reshape(n, -1)[:, 0]
            out = {
                "sim_data_projected": window_subsample(
                    projected, j, sub["windows"], sub["max_particles"], sub["pad_value"], rng
                ),
                "vcirc_kms": vcirc,
            }
        # Ibata ancillary observables (only present when requested): vterm_kms (n, n_l, 1),
        # sigma_z (n, 1), rho_z (n, n_z, 1). Each is a deterministic function of the shared
        # potential; the noisy "observed" counterparts are added later by the augmentation chain.
        # (M200, c_v') halo: store the per-row densityNorm / scaleRadius AGAMA received. These are
        # derived diagnostics (not inferred); the identity rho/a params stay fixed constants.
        halo_list = [r[3] for r in results]
        if halo_list[0] is not None:
            derived = np.asarray(halo_list, dtype=float)  # (n, 2): [densityNorm, scaleRadius]
            out["rho_TwoPowerTriaxial_halo_derived"] = derived[:, 0:1]
            out["a_TwoPowerTriaxial_halo_derived"] = derived[:, 1:2]
        # Restricted N-body only: the present-day bound mass of the remnant. ``m_progenitor`` is the
        # mass at t = -t_end, so this is the quantity that should match the observed cluster mass —
        # a diagnostic (and a potential constraint), never an inferred parameter, so the adapter
        # drops it like the derived m200_c rho/a.
        bound_list = [r[4] for r in results]
        if bound_list[0] is not None:
            out["m_bound_final"] = np.asarray(bound_list, dtype=float).reshape(-1, 1)
        anc_list = [r[2] for r in results]
        if anc_list[0]:  # empty dict when no ancillary observables were requested
            for key in ("vterm_kms", "rho_z"):
                if key in anc_list[0]:
                    out[key] = np.stack([a[key] for a in anc_list], axis=0)[..., None]
            if "sigma_z" in anc_list[0]:
                out["sigma_z"] = np.stack([a["sigma_z"] for a in anc_list], axis=0)  # (n, 1)
        return out

    def sample_compositional(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """One shared global draw per dataset, one stream realization per target stream.

        Globals ``(n, 1)``; ``j`` / locals ``(n, n_streams, 1)``; stream observables
        ``(n, n_streams, n_particles, 6)``. The rotation curve depends only on the shared
        potential, so it stays ``(n, n_radii, 1)``.
        """
        def draw(k: int, r: np.random.Generator) -> Dict[str, np.ndarray]:
            return sample_stream_prior_shared_global(
                self._priors_global, self._priors_local, self.target_streams, k, r
            )

        if self._vcirc_rejection is None:
            draws = draw(n, rng)
        else:
            draws = self._rejection_sample(n, rng, draw)
        m = len(self.target_streams)

        # Flatten (dataset, stream) pairs into rows and reuse the row-parallel simulate().
        flat: Dict[str, np.ndarray] = {}
        for key, arr in draws.items():
            if arr.shape[:2] == (n, m):
                flat[key] = arr.reshape(n * m, 1)
            else:  # global: (n, 1) -> repeat per stream
                flat[key] = np.repeat(arr, m, axis=0)

        sims = self.simulate(flat, rng)

        out = dict(draws)
        if "sim_data_carthesian" in sims:
            out["sim_data_carthesian"] = sims["sim_data_carthesian"].reshape(
                n, m, self._n_particles, 6
            )
        # The stored particle count is n_particles, or the cap under `store_window_subsample`.
        out["sim_data_projected"] = sims["sim_data_projected"].reshape(n, m, -1, 6)
        # Identical for the m streams of a dataset (shared potential): keep one per dataset.
        out["vcirc_kms"] = sims["vcirc_kms"].reshape(n, m, -1, 1)[:, 0]
        # Ibata ancillary observables also depend only on the shared potential -> one per dataset.
        for key in ("vterm_kms", "rho_z"):  # (n*m, bins, 1) -> (n, bins, 1)
            if key in sims:
                out[key] = sims[key].reshape(n, m, -1, 1)[:, 0]
        if "sigma_z" in sims:  # (n*m, 1) -> (n, 1)
            out["sigma_z"] = sims["sigma_z"].reshape(n, m, 1)[:, 0]
        # Derived (M200, c_v') halo params depend only on the shared potential -> one per group.
        for key in ("rho_TwoPowerTriaxial_halo_derived", "a_TwoPowerTriaxial_halo_derived"):
            if key in sims:  # (n*m, 1) -> (n, 1)
                out[key] = sims[key].reshape(n, m, 1)[:, 0]
        # The bound remnant mass belongs to each stream's own progenitor, so it stays per-member
        # (n, m, 1) like the locals — unlike everything above, which is a property of the shared
        # potential.
        if "m_bound_final" in sims:
            out["m_bound_final"] = sims["m_bound_final"].reshape(n, m, 1)
        return out
