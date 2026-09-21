"""Particle-spray stellar streams on **gala** (CPU, joblib) in a MilkyWayPotential2022-style Galaxy.

A controlled twin of the AGAMA spray simulators with a different forward-model stack:

* **Potential** — gala's ``MilkyWayPotential2022`` component family: fixed Hernquist bulge + nucleus
  (the MW2022 values), a free ``MN3ExponentialDiskPotential`` (``m_disk``, ``h_R_disk``,
  ``h_z_disk``) and an NFW halo instantiated from a virial mass and concentration
  (``log10_M200_halo``, ``c200_halo`` — gala's ``NFWPotential.from_M200_c`` algebra, Planck18
  critical density) with **flattening in the potential** (gala's ``c`` axis ratio,
  ``Phi(sqrt(R^2 + z^2/c^2))``).
* **Halo shape prior on the DENSITY axis ratio** ``q_rho_halo`` (comparable to every AGAMA
  generation, where ``q_TwoPowerTriaxial_halo`` flattens the density). Each row's potential
  flattening ``c_phi`` is derived by inverting the Poisson map of the analytic flattened NFW at a
  reference radius (``q_ref_r_kpc``): ``q_rho = z*/r_ref`` with ``rho(0, 0, z*) = rho(r_ref, 0, 0)``.
  Stored as ``c_phi_halo_derived``. The potential-flattened NFW has NEGATIVE density on the pole
  for ``c_phi < ~0.89`` (``q_rho < ~0.78``), but only far out (z > 36 kpc at c_phi = 0.75 for
  r_s = 15.6 kpc) — accepted; the radius where it turns negative is stored as
  ``halo_rho_neg_r_kpc_derived`` (inf when the density is positive everywhere).
* **Stream** — Chen, Gnedin & Li (2024) release recipe, gala's own ``ChenStreamDF`` inside
  ``MockStreamGenerator`` with a moving Plummer progenitor potential; linear mass loss from
  ``m_progenitor`` (initial) to ``m_progenitor_final`` through gala's time-varying ``prog_mass``.
* **Frame** — astropy's default ``Galactocentric`` for both the progenitor's observed-ICRS ->
  Galactocentric conversion and the particle projection (:func:`stream_common.sky_projection` with
  ``frames=None``). Unlike the v4 AGAMA configs the solar phase-space frame is NOT varied.

gala is imported **only inside the joblib worker** (a fresh loky process), never in the main
process — the pipeline's main process carries JAX/BayesFlow, and keeping C-extension stacks apart is
the conservative rule this project follows (galpy has a known static-TLS clash with JAX).

Everything else — prior declaration, ``sample_prior``, ``sample_compositional``, ``simulate``'s
dispatch, the in-window storage cap and the custom rotation-curve grid — is inherited from
:class:`stream_agama.AgamaStreamSimulator`; only the row worker is swapped (``_row_jobs``).
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from hydrabflow.registry import register_simulator
from hydrabflow.simulators.stream_agama import AgamaStreamSimulator, _mass_track
from hydrabflow.simulators.stream_common import G_KPC_KMS2_MSUN
from hydrabflow.utils.quiet import quiet_worker

# Planck18 critical density at z=0 in Msun/kpc^3 (astropy.cosmology.Planck18.critical_density(0)).
# gala's NFWPotential.from_M200_c uses astropy's default cosmology, which is Planck18 — pinning the
# number here keeps the (M200, c) -> (m, r_s) map reproducible without importing astropy.cosmology in
# the worker. 1.6 % below the H0 = 70.4 value the AGAMA m200_c halo uses.
RHO_CRIT_PLANCK18_MSUN_KPC3 = 127.0528153974983

# gala MilkyWayPotential2022 components that stay FIXED here (special.py `_setup_mwp_2022`).
FIXED_BULGE = {"m": 5.0e9, "c": 1.0}  # HernquistPotential, Msun / kpc
FIXED_NUCLEUS = {"m": 1.8142e9, "c": 0.0688867}
MW22_DISK = {"m": 4.7717e10, "h_R": 2.6, "h_z": 0.3}  # MN3ExponentialDiskPotential; free here
MW22_HALO = {"m": 5.5427e11, "r_s": 15.626}  # spherical NFW; == M200 9.6e11, c200 13.3 (Planck18)

DERIVED_KEYS = (
    "c_phi_halo_derived",
    "halo_rho_neg_r_kpc_derived",
    "m_nfw_halo_derived",
    "r_s_halo_derived",
)


# --------------------------------------------------------------------------------------------------
# Pure-numpy halo helpers (importable and testable without gala)
# --------------------------------------------------------------------------------------------------
def nfw_m_rs_from_m200_c(
    m200: float, c200: float, rho_c: float = RHO_CRIT_PLANCK18_MSUN_KPC3
) -> tuple[float, float]:
    """gala's ``NFWPotential.from_M200_c`` algebra: ``R200 = cbrt(M200 / (200 rho_c) / (4 pi / 3))``,
    ``r_s = R200 / c``, scale mass ``m = M200 / (ln(1 + c) - c / (1 + c))``. Returns ``(m [Msun],
    r_s [kpc])``."""
    r200 = np.cbrt(float(m200) / (200.0 * float(rho_c)) / (4.0 / 3.0 * np.pi))
    r_s = r200 / float(c200)
    a_nfw = np.log1p(c200) - c200 / (1.0 + c200)
    return float(m200) / float(a_nfw), float(r_s)


def flattened_nfw_density(
    R, z, m: float, r_s: float, c_phi: float, G: float = G_KPC_KMS2_MSUN
) -> np.ndarray:
    """Density ``Laplacian(Phi) / (4 pi G)`` of gala's flattened NFW potential
    ``Phi = -G m / r_s * ln(1 + u) / u``, ``u = s / r_s``, ``s = sqrt(R^2 + z^2 / c_phi^2)``.

    Analytic: with ``f(s) = Phi``, ``Lap f = f''(s) |grad s|^2 + f'(s) Lap s`` where
    ``|grad s|^2 = (R^2 + z^2 / c^4) / s^2`` and ``Lap s = (2 + 1 / c^2) / s - (R^2 + z^2 / c^4) / s^3``.
    Vectorised over ``R`` and ``z`` (broadcast). Negative values are real: flattening the potential
    does not guarantee a positive mass distribution (gala's NFWPotential docstring).
    """
    R = np.asarray(R, dtype=float)
    z = np.asarray(z, dtype=float)
    c2 = float(c_phi) ** 2
    s = np.sqrt(R * R + z * z / c2)
    u = s / r_s
    # g(u) = ln(1+u)/u and its derivatives
    lg = np.log1p(u)
    g1 = 1.0 / (u * (1.0 + u)) - lg / u**2
    g2 = -(1.0 + 2.0 * u) / (u**2 * (1.0 + u) ** 2) - 1.0 / (u**2 * (1.0 + u)) + 2.0 * lg / u**3
    A = -G * m / r_s
    f1 = A * g1 / r_s
    f2 = A * g2 / r_s**2
    w = R * R + z * z / c2**2  # R^2 + z^2 / c^4
    grad_s2 = w / s**2
    lap_s = (2.0 + 1.0 / c2) / s - w / s**3
    return (f2 * grad_s2 + f1 * lap_s) / (4.0 * np.pi * G)


def halo_rho_negative_radius(c_phi: float, r_s: float, r_max: float = 1000.0) -> float:
    """Smallest radius on the pole where the flattened NFW density is negative; ``inf`` if none in
    ``(0, r_max]``. Amplitude-free (``m = 1``, ``G = 1``)."""
    from scipy.optimize import brentq

    z = np.geomspace(1e-3 * r_s, r_max, 2000)
    rho = flattened_nfw_density(0.0, z, 1.0, r_s, c_phi, G=1.0)
    neg = np.flatnonzero(rho < 0)
    if neg.size == 0:
        return float("inf")
    i = int(neg[0])
    if i == 0:
        return float(z[0])
    f = lambda zz: float(flattened_nfw_density(0.0, zz, 1.0, r_s, c_phi, G=1.0))  # noqa: E731
    return float(brentq(f, z[i - 1], z[i]))


def q_rho_from_c_phi(c_phi: float, r_s: float, r_ref: float) -> float:
    """Density axis ratio ``z* / r_ref`` of the isodensity surface through ``(r_ref, 0, 0)``.

    ``z*`` solves ``rho(0, 0, z*) = rho(r_ref, 0, 0)`` on the pole; the overall amplitude cancels.
    The pole density is monotone in ``z`` out to where it turns negative, so the bracket is
    ``[1e-3 r_ref, min(z_neg, 50 r_ref)]``."""
    from scipy.optimize import brentq

    target = float(flattened_nfw_density(r_ref, 0.0, 1.0, r_s, c_phi, G=1.0))
    z_hi = min(halo_rho_negative_radius(c_phi, r_s, r_max=50.0 * r_ref) * 0.999, 50.0 * r_ref)
    f = lambda zz: float(flattened_nfw_density(0.0, zz, 1.0, r_s, c_phi, G=1.0)) - target  # noqa: E731
    return float(brentq(f, 1e-3 * r_ref, z_hi)) / float(r_ref)


def c_phi_from_q_rho(
    q_rho: float, r_s: float, r_ref: float, bracket: tuple[float, float] = (0.6, 1.6)
) -> float:
    """Invert :func:`q_rho_from_c_phi` for the potential flattening giving density axis ratio
    ``q_rho`` at ``r_ref``. Raises ``ValueError`` (with the inputs) if ``bracket`` does not straddle
    the root, so a too-narrow bracket is reported rather than silently clipped."""
    from scipy.optimize import brentq

    lo, hi = (float(bracket[0]), float(bracket[1]))
    f = lambda c: q_rho_from_c_phi(c, r_s, r_ref) - float(q_rho)  # noqa: E731
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError(
            f"c_phi_from_q_rho: bracket {bracket} does not straddle q_rho={q_rho:.3f} at "
            f"r_s={r_s:.2f} kpc, r_ref={r_ref:.1f} kpc (q_rho at bracket ends: {flo + q_rho:.3f}, "
            f"{fhi + q_rho:.3f}); widen `c_phi_bracket`."
        )
    return float(brentq(f, lo, hi, xtol=1e-8))


# --------------------------------------------------------------------------------------------------
# Worker (gala imported here only)
# --------------------------------------------------------------------------------------------------
def _build_potential(gp, u, galactic, p: Mapping[str, float], opts: Mapping):
    """The composite potential of one row + its derived diagnostics dict."""
    m_nfw, r_s = nfw_m_rs_from_m200_c(10.0 ** p["log10_M200_halo"], p["c200_halo"], opts["rho_c"])
    c_phi = c_phi_from_q_rho(p["q_rho_halo"], r_s, opts["q_ref_r_kpc"], opts["c_phi_bracket"])
    pot = gp.CCompositePotential()
    pot["disk"] = gp.MN3ExponentialDiskPotential(
        m=p["m_disk"] * u.Msun, h_R=p["h_R_disk"] * u.kpc, h_z=p["h_z_disk"] * u.kpc, units=galactic
    )
    pot["bulge"] = gp.HernquistPotential(m=p["m_bulge"] * u.Msun, c=p["c_bulge"] * u.kpc, units=galactic)
    pot["nucleus"] = gp.HernquistPotential(
        m=p["m_nucleus"] * u.Msun, c=p["c_nucleus"] * u.kpc, units=galactic
    )
    pot["halo"] = gp.NFWPotential(
        m=m_nfw * u.Msun, r_s=r_s * u.kpc, a=1.0, b=1.0, c=c_phi, units=galactic
    )
    derived = {
        "c_phi_halo_derived": c_phi,
        "halo_rho_neg_r_kpc_derived": halo_rho_negative_radius(c_phi, r_s),
        "m_nfw_halo_derived": m_nfw,
        "r_s_halo_derived": r_s,
    }
    return pot, derived


def _prog_w0(coord, u, gd, p: Mapping[str, float]):
    """Observed ICRS progenitor -> Galactocentric phase-space position (astropy default frame, the
    same frame :func:`stream_common.sky_projection` uses for the particles)."""
    c = coord.SkyCoord(
        ra=p["ra"] * u.deg,
        dec=p["dec"] * u.deg,
        distance=p["r"] * u.kpc,
        pm_ra_cosdec=p["mu_ra_cosdec"] * u.mas / u.yr,
        pm_dec=p["mu_dec"] * u.mas / u.yr,
        radial_velocity=p["vr"] * u.km / u.s,
        frame="icrs",
    )
    gc = c.transform_to(coord.Galactocentric())
    return gd.PhaseSpacePosition(gc.data)


def _vcirc_gala(pot, u, obs_r: np.ndarray) -> np.ndarray:
    """Circular speed [km/s] in the plane at the observed radii. Independent of the potential
    flattening (``Phi(R, 0) = Phi_spherical(R)``)."""
    q = np.zeros((3, len(obs_r)))
    q[0] = np.asarray(obs_r, dtype=float)
    return np.asarray(pot.circular_velocity(q * u.kpc).to(u.km / u.s).value, dtype=float)


@quiet_worker
def _simulate_one_gala(
    p: Dict[str, float], n_particles: int, obs_r: np.ndarray, seed: int, opts: Mapping
):
    """joblib worker: one Chen+2024 spray stream in the row's composite potential + the rotation
    curve of that potential. Returns the 6-tuple ``simulate`` expects (see
    ``AgamaStreamSimulator._row_jobs``): ``(xv (n_particles, 6) Galactocentric kpc & km/s, vcirc
    (n_radii,) km/s, {}, derived dict, None, frame NaNs)``. A failed stream integration yields an
    all-NaN ``xv`` (the row is dropped by ``drop_nan``) while the potential-only outputs survive."""
    import warnings

    warnings.simplefilter("ignore")  # gala's GalaFutureWarning / astropy unit chatter per row
    import astropy.coordinates as coord
    import astropy.units as u
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.units import galactic

    rs = np.random.RandomState(int(seed) % 2**32)  # ChenStreamDF draws via `random_state`
    pot, derived = _build_potential(gp, u, galactic, p, opts)
    vcirc = _vcirc_gala(pot, u, obs_r)

    n_steps = int(opts["n_steps"])
    npr = int(opts["n_particles_per_release"])
    t_end_myr = float(p["t_end"]) * 1e3
    xv = np.full((n_particles, 6), np.nan)
    try:
        w0 = _prog_w0(coord, u, gd, p)
        # Progenitor mass at each orbit time (n_steps + 1 knots, past -> present, matching the
        # reversed orbit gala hands the DF): linear decline from the initial to the present-day mass.
        m_i, m_f = float(p["m_progenitor"]), float(p["m_progenitor_final"])
        if opts["mass_loss"] is None:
            prog_mass = m_i * u.Msun
        else:
            t_grid = -t_end_myr + (t_end_myr / n_steps) * np.arange(n_steps + 1)
            prog_mass = _mass_track(m_i, m_f, t_grid, t_end_myr, opts["mass_loss"]) * u.Msun
        # Fixed-mass Plummer well the released stars feel (gala cannot vary it in time): the
        # time-average of the mass track.
        m_pot = 0.5 * (m_i + m_f) if (opts["mass_loss"] is not None and m_f < m_i) else m_i
        prog_pot = gp.PlummerPotential(m=m_pot * u.Msun, b=p["a_progenitor"] * u.pc, units=galactic)
        gen = gd.MockStreamGenerator(
            gd.ChenStreamDF(random_state=rs), gp.Hamiltonian(pot), progenitor_potential=prog_pot
        )
        # Integrator: gala steps ALL released particles in one adaptive dop853 call per time step
        # (mockstream.pyx), so a single particle diving through the 30 pc Plummer core collapses the
        # shared step and the whole row fails with "Integration failed with code -4" (measured on
        # 1e4-star rows: ~15-40 % of NGC3201/M68 rows, the stock MilkyWayPotential2022 included).
        # Fixed-step leapfrog (dt = t_end / n_steps ~ 0.4-2 Myr, ~100+ steps per orbit; the Plummer
        # dynamical time is ~8 Myr) never fails and agrees with dop853 where both run: per-particle
        # |dx| 6 pc median / 40 pc 90th pct after 4 Gyr, in-window track medians within 0.06 deg.
        if opts["integrator"] == "leapfrog":
            from gala.integrate import LeapfrogIntegrator as Integrator
        else:
            from gala.integrate import DOPRI853Integrator as Integrator
        stream, _ = gen.run(
            w0, prog_mass, dt=-(t_end_myr / n_steps) * u.Myr, n_steps=n_steps,
            n_particles=npr, release_every=1, progress=False, Integrator=Integrator,
        )
        pos = stream.xyz.to(u.kpc).value  # (3, N)
        vel = stream.v_xyz.to(u.km / u.s).value
        out = np.column_stack([pos.T, vel.T])
        if len(out) != n_particles:
            raise RuntimeError(
                f"gala released {len(out)} stars but n_particles={n_particles}: gala emits "
                f"2 * (n_steps + 1) * n_particles_per_release, so set n_steps = n_particles / "
                f"(2 * n_particles_per_release) - 1."
            )
        # The Chen DF draws the release radius r ~ N(1.6, 0.35) r_j with no floor, so ~2e-6 of
        # draws are negative and give a NaN escape speed (df.pyx `sqrt(2 G m / Dr)`); leapfrog then
        # carries that NaN particle to the end. Replace up to 0.1 % such stars by re-drawing from the
        # finite ones (a negligible duplication) instead of losing the whole row.
        bad = ~np.isfinite(out).all(axis=1)
        if bad.any():
            if bad.sum() > max(1, n_particles // 1000):
                raise RuntimeError(f"{int(bad.sum())} non-finite stream particles")
            good = np.flatnonzero(~bad)
            out[bad] = out[rs.choice(good, size=int(bad.sum()), replace=False)]
        xv = out
    except Exception:
        if opts.get("raise_errors"):
            raise
        # row -> NaN (drop_nan removes it); the potential-only outputs above survive
    return xv, vcirc, {}, derived, None, np.full(5, np.nan)


# --------------------------------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------------------------------
@register_simulator("stream_gala")
class GalaStreamSimulator(AgamaStreamSimulator):
    """gala Chen+2024 particle spray in a MilkyWayPotential2022-style composite (see module doc).

    Reuses the AGAMA class's prior declaration, ``sample_prior``, ``sample_compositional`` and
    ``simulate`` dispatch; swaps only the row worker. Extra ``params``:

    * ``n_steps`` (4999) — progenitor orbit steps; gala releases ``2 * (n_steps + 1) *
      n_particles_per_release`` stars, which MUST equal ``n_particles`` (checked).
    * ``n_particles_per_release`` (1), ``q_ref_r_kpc`` (15.0), ``c_phi_bracket`` ([0.6, 1.6]),
      ``rho_c_msun_kpc3`` (null -> Planck18, as gala), ``mass_loss`` (``linear`` | ``exponential`` |
      null = fixed mass; a law needs ``m_progenitor_final`` per stream), ``integrator``
      (``leapfrog`` default — see the worker for why not gala's default dop853 — | ``dopri853``).

    ``vcirc_rejection`` and a marginalized solar frame are not supported on this class.
    """

    @property
    def _gala_opts(self) -> Dict:
        n_steps = int(self.params.get("n_steps", 4999))
        npr = int(self.params.get("n_particles_per_release", 1))
        expected = 2 * (n_steps + 1) * npr
        if expected != self._n_particles:
            raise ValueError(
                f"stream_gala: n_particles={self._n_particles} but gala releases 2 * (n_steps + 1) "
                f"* n_particles_per_release = {expected}; set n_steps={self._n_particles // (2 * npr) - 1}."
            )
        if self.params.get("vcirc_rejection") is not None:
            raise ValueError("stream_gala does not support `vcirc_rejection`.")
        if any(k in self._priors_global for k in ("R0_Sun", "U_Sun", "V_Sun", "W_Sun")):
            raise ValueError(
                "stream_gala projects through astropy's default Galactocentric frame; solar "
                "phase-space priors (R0_Sun/U_Sun/V_Sun/W_Sun) are not supported here."
            )
        mass_loss = self.params.get("mass_loss")
        rho_c = self.params.get("rho_c_msun_kpc3")
        bracket = self.params.get("c_phi_bracket", [0.6, 1.6])
        return {
            "n_steps": n_steps,
            "n_particles_per_release": npr,
            "q_ref_r_kpc": float(self.params.get("q_ref_r_kpc", 15.0)),
            "c_phi_bracket": (float(bracket[0]), float(bracket[1])),
            "rho_c": RHO_CRIT_PLANCK18_MSUN_KPC3 if rho_c is None else float(rho_c),
            "mass_loss": None if mass_loss is None else str(mass_loss),
            "integrator": str(self.params.get("integrator", "leapfrog")),  # leapfrog | dopri853
            "raise_errors": bool(self.params.get("raise_errors", False)),  # debugging aid
        }

    def _row_jobs(self, rows, seeds):
        from joblib import delayed

        opts = self._gala_opts
        return [
            delayed(_simulate_one_gala)(row, self._n_particles, self.obs_r_kpc, int(seed), opts)
            for row, seed in zip(rows, seeds)
        ]
