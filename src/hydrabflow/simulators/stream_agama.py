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
from hydrabflow.simulators.registry import register_simulator
from hydrabflow.simulators.stream_common import (
    OBS_R_KPC,
    OBS_SIGMA_VC,
    OBS_VC_KMS,
    RHO_Z_KPC,
    VTERM_L_DEG,
    extended_rotation_curve,
    inferred_names,
    sample_stream_prior,
    sample_stream_prior_shared_global,
    sky_projection,
    surface_density,
    terminal_velocity,
    vertical_density_profile,
)

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


def _halo_params(p: Mapping[str, float], r_t: float) -> dict:
    return dict(
        type="Spheroid",
        scaleRadius=p["a_TwoPowerTriaxial_halo"],
        densityNorm=p["rho_TwoPowerTriaxial_halo"],
        gamma=p["gamma_TwoPowerTriaxial_halo"],
        alpha=1,
        beta=p["beta_TwoPowerTriaxial_halo"],
        cutoffStrength=2,
        outerCutoffRadius=r_t,
        axisRatioY=1.0,
        axisRatioZ=p["q_TwoPowerTriaxial_halo"],
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
    components = [BULGE_PARAMS]
    if cfg["gas_disks"]:
        components += [GAS_HI_PARAMS, GAS_H2_PARAMS]
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
    rj = (agama.G * mass_sat / (Omega**2 - d2Phi_dr2)) ** (1.0 / 3)
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
    with np.errstate(invalid="ignore"):
        Dv = s[:, 3] * np.sqrt(2.0 * G * mass_sat / Dr)  # local escape speed at Dr
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


def _spray_stream(
    agama, pot_host, posvel_sat: np.ndarray, mass_sat: float, radius_sat: float,
    time_total: float, num_particles: int, rng: np.random.Generator,
    method: str = "fardal",
) -> np.ndarray:
    """Particle-spray stream including the progenitor's own (moving Plummer) potential.

    Invalid seeds (undefined Jacobi radius where ``Omega^2 - d2Phi/dr2 < 0``) stay NaN so the
    output always has shape ``(num_particles, 6)``; downstream NaN cleaning drops those rows.
    """
    N = num_particles // 2
    time_sat, orbit_sat = agama.orbit(
        potential=pot_host, ic=posvel_sat, time=-time_total, trajsize=N + 1, accuracy=1e-10
    )
    time_sat = time_sat[1:][::-1]
    orbit_sat = orbit_sat[1:][::-1]

    rj, vj, R = _rj_vj_R(agama, pot_host, orbit_sat, mass_sat)
    if method == "chen":
        ic_stream = _ic_chen_spray(orbit_sat, rj, R, rng, mass_sat, agama.G)
    else:
        ic_stream = _ic_particle_spray(orbit_sat, rj, vj, R, rng)
    time_seed = np.repeat(time_sat, 2)

    pot_sat = agama.Potential(
        type="Plummer",
        mass=mass_sat,
        scaleRadius=radius_sat,
        center=np.column_stack([time_sat, orbit_sat]),
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


def _vcirc(pot_host, obs_r: np.ndarray) -> np.ndarray:
    """Model circular velocity [km/s] at the observed radii; NaN where v^2 < 0."""
    points = np.column_stack((obs_r, np.zeros_like(obs_r), np.zeros_like(obs_r)))
    v2 = -obs_r * pot_host.force(points)[:, 0]
    return np.sqrt(np.where(v2 > 0, v2, np.nan))


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
    agama, pot_host, p: Dict[str, float], pot_cfg: Mapping | None, spec: Mapping | None
) -> Dict[str, np.ndarray]:
    """Ibata (2023) potential-derived observables for one row, per ``spec`` (an empty/None spec
    computes nothing). ``spec`` carries the requested observable names + their fixed grids:
    ``{"names": [...], "l_deg": array, "z_kpc": array}``. Deterministic force/density evaluations
    on the assembled potential — no stream simulation."""
    if not spec or not spec.get("names"):
        return {}
    names = set(spec["names"])
    out: Dict[str, np.ndarray] = {}
    if "vterm" in names:
        out["vterm_kms"] = terminal_velocity(pot_host, spec["l_deg"]).astype(float)
    if "sigma_z" in names:
        out["sigma_z"] = np.array([surface_density(pot_host)], dtype=float)
    if "rho_z" in names:
        disk_pot = _stellar_disk_potential(agama, p, pot_cfg)
        out["rho_z"] = vertical_density_profile(disk_pot, spec["z_kpc"]).astype(float)
    return out


def _simulate_one(
    p: Dict[str, float], n_particles: int, obs_r: np.ndarray, seed: int,
    spray_method: str = "fardal", pot_cfg: Mapping | None = None,
    ancillary: Mapping | None = None,
):
    """joblib worker: one stream realization + the rotation curve of its potential (+ the optional
    Ibata ancillary observables of that potential)."""
    agama = _agama()
    rng = np.random.default_rng(seed)
    # getUnits()['time'] is a float [Myr] normally, but an astropy Quantity once astropy has
    # been imported in this process (agama >= 1.0.157) — take .value in that case.
    tu = agama.getUnits()["time"]
    time_unit_gyr = float(getattr(tu, "value", tu)) / 1e3

    pot_host = _host_potential(agama, p, pot_cfg)

    l0, b0, pml0, pmb0 = agama.transformCelestialCoords(
        agama.fromICRStoGalactic,
        p["ra"] * np.pi / 180,
        p["dec"] * np.pi / 180,
        p["mu_ra_cosdec"],
        p["mu_dec"],
    )
    posvel_sat = np.array(
        agama.getGalactocentricFromGalactic(l0, b0, p["r"], pml0 * 4.74, pmb0 * 4.74, p["vr"])
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
    )
    anc = _ancillary_observables(agama, pot_host, p, pot_cfg, ancillary)
    return xv, _vcirc(pot_host, obs_r), anc


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
    def _n_workers(self) -> int:
        return int(self.params.get("n_workers", 8))

    @property
    def _obs_r_split(self) -> float:
        """Split radius for the extended (Zhou u Huang) rotation-curve grid."""
        return float(self.params.get("obs_r_split_kpc", float(OBS_R_KPC.max())))

    @property
    def obs_r_kpc(self) -> np.ndarray:
        """Radii the model rotation curve is evaluated on (also the ``vcirc_kms`` grid).

        ``obs_r_grid: extended`` -> Zhou below ``obs_r_split_kpc`` u Huang beyond (50 radii,
        the single source of truth being ``stream_common.extended_rotation_curve``); an
        explicit ``obs_r_kpc`` list overrides; otherwise the Zhou grid.
        """
        if str(self.params.get("obs_r_grid", "")) == "extended":
            return extended_rotation_curve(self._obs_r_split)[0]
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
        if str(self.params.get("obs_r_grid", "")) == "extended":
            return extended_rotation_curve(self._obs_r_split)[2]
        return np.asarray(self.params.get("obs_sigma_vc", OBS_SIGMA_VC), dtype=float)

    @property
    def obs_vc_kms(self) -> np.ndarray:
        """Observed Milky Way circular velocity aligned with ``obs_r_kpc`` — the fixed curve
        ``attach_observed_vcirc`` supplies to real-data evaluation (simulated data carry their own
        model curve). ``obs_r_grid: extended`` -> Zhou below the split u Huang beyond; otherwise
        the Zhou grid. Mirrors :meth:`obs_r_kpc` / :meth:`obs_sigma_vc`."""
        if str(self.params.get("obs_r_grid", "")) == "extended":
            return extended_rotation_curve(self._obs_r_split)[1]
        return np.asarray(self.params.get("obs_vc_kms", OBS_VC_KMS), dtype=float)

    # ------------------------------------------------------------------------------------- #
    # Potential model + Ibata ancillary observables (config-driven; legacy defaults are no-ops)
    # ------------------------------------------------------------------------------------- #

    @property
    def _pot_cfg(self) -> dict:
        """Host-potential configuration threaded to the joblib workers. Legacy default (all keys
        absent) reproduces the original bulge + untruncated halo + one isothermal disk; the Ibata
        sim config sets ``halo_r_t_kpc`` / ``gas_disks`` / ``thick_disk`` / ``disk_vertical``."""
        return dict(
            halo_r_t_kpc=float(self.params.get("halo_r_t_kpc", float("inf"))),
            gas_disks=bool(self.params.get("gas_disks", False)),
            thick_disk=bool(self.params.get("thick_disk", False)),
            disk_vertical=str(self.params.get("disk_vertical", "isothermal")),
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
    def global_parameter_names(self) -> list[str]:
        return inferred_names(self._priors_global)

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

    def simulate(
        self, params: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        from joblib import Parallel, delayed

        n = len(np.asarray(next(iter(params.values()))))
        rows = [
            {k: float(np.asarray(v).reshape(n, -1)[i, 0]) for k, v in params.items()}
            for i in range(n)
        ]
        seeds = rng.integers(0, 2**31 - 1, size=n)
        spray_method = str(self.params.get("spray_method", "fardal"))
        pot_cfg = self._pot_cfg
        ancillary = self._ancillary_spec

        # batch_size=1: forward-model rows have highly variable cost (restricted N-body rows,
        # especially the long t_end streams, run far longer than spray rows). joblib's default
        # batch_size='auto' grows the batch after fast early rows and then dispatches a few large
        # batches to a handful of workers, starving the rest and serializing the slow tail. One
        # row per dispatch keeps all n_workers busy; the per-task overhead is negligible next to a
        # multi-second row.
        results = Parallel(n_jobs=self._n_workers, batch_size=1)(
            delayed(_simulate_one)(
                row, self._n_particles, self.obs_r_kpc, int(seed), spray_method, pot_cfg, ancillary
            )
            for row, seed in zip(rows, seeds)
        )
        xv = np.stack([r[0] for r in results], axis=0)  # (n, n_particles, 6)
        vcirc = np.stack([r[1] for r in results], axis=0)[..., None]  # (n, n_radii, 1)

        out = {
            "sim_data_carthesian": xv,
            "sim_data_projected": sky_projection(xv),
            "vcirc_kms": vcirc,
        }
        # Ibata ancillary observables (only present when requested): vterm_kms (n, n_l, 1),
        # sigma_z (n, 1), rho_z (n, n_z, 1). Each is a deterministic function of the shared
        # potential; the noisy "observed" counterparts are added later by the augmentation chain.
        anc_list = [r[2] for r in results]
        if ancillary:
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
        out["sim_data_carthesian"] = sims["sim_data_carthesian"].reshape(
            n, m, self._n_particles, 6
        )
        out["sim_data_projected"] = sims["sim_data_projected"].reshape(
            n, m, self._n_particles, 6
        )
        # Identical for the m streams of a dataset (shared potential): keep one per dataset.
        out["vcirc_kms"] = sims["vcirc_kms"].reshape(n, m, -1, 1)[:, 0]
        # Ibata ancillary observables also depend only on the shared potential -> one per dataset.
        for key in ("vterm_kms", "rho_z"):  # (n*m, bins, 1) -> (n, bins, 1)
            if key in sims:
                out[key] = sims[key].reshape(n, m, -1, 1)[:, 0]
        if "sigma_z" in sims:  # (n*m, 1) -> (n, 1)
            out["sigma_z"] = sims["sigma_z"].reshape(n, m, 1)[:, 0]
        return out
