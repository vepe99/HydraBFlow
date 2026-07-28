"""Tests for the (M200, c_v') halo reparameterization (McMillan 2017; stream_agama._halo_params_m200c).

Validates the overdensity concentration conversion and that the (M200, c_v', gamma) -> (r_h,
densityNorm) chain reproduces McMillan (2017) Table 3, that enclosedMass(r200) == M200 exactly, and
that _host_potential dispatches to the m200_c builder (and matches the equivalent rho_a halo).
"""
from __future__ import annotations

import numpy as np
import pytest

agama = pytest.importorskip("agama")

from hydrabflow.simulators.stream_agama import (  # noqa: E402
    AgamaStreamSimulator,
    _halo_params,
    _halo_params_m200c,
    _host_potential,
)
from hydrabflow.simulators.stream_common import convert_concentration  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _units():
    agama.setUnits(mass=1, length=1, velocity=1)


# McMillan (2017) cosmology / overdensity conventions used by the m200_c builder.
_CFG = dict(
    halo_r_t_kpc=1000.0,
    halo_H0_kms_mpc=70.4,
    halo_Delta_mass=200.0,
    halo_Delta_c=94.0,
)


def _rho_crit():
    H0 = _CFG["halo_H0_kms_mpc"] / 1000.0
    return 3.0 * H0**2 / (8.0 * np.pi * agama.G)


def test_convert_concentration_roundtrip_and_monotonic():
    # converting there-and-back is the identity
    c1 = 15.4
    c200 = convert_concentration(c1, 94.0, 200.0)
    assert convert_concentration(c200, 200.0, 94.0) == pytest.approx(c1, rel=1e-4)
    # lower overdensity -> larger enclosed radius -> larger concentration
    assert c200 < c1
    # McMillan best fit: c_v'(94)=15.4 maps to c200 ~ 11.4
    assert c200 == pytest.approx(11.4, abs=0.2)


def test_reproduces_mcmillan_table3():
    """M200=1.3e12, c_v'=15.4, gamma=1 -> r_h=19.6 kpc, rho_0=8.54e6 (McMillan Table 3)."""
    p = {
        "log10_M200_TwoPowerTriaxial_halo": np.log10(1.30e12),
        "ln_cvprime_TwoPowerTriaxial_halo": np.log(15.4),
        "gamma_TwoPowerTriaxial_halo": 1.0,
        "beta_TwoPowerTriaxial_halo": 3.0,
        "q_TwoPowerTriaxial_halo": 1.0,
    }
    halo = _halo_params_m200c(agama, p, _CFG)
    assert halo["scaleRadius"] == pytest.approx(19.6, abs=0.3)
    assert halo["densityNorm"] == pytest.approx(8.54e6, rel=0.02)


@pytest.mark.parametrize("gamma", [-1.5, 0.0, 1.0, 1.5])
@pytest.mark.parametrize("log10M200", [np.log10(0.5e12), np.log10(1.3e12), np.log10(2.5e12)])
def test_enclosed_mass_equals_M200(gamma, log10M200):
    """The unit-norm densityNorm solve makes enclosedMass(r200) == M200 for any gamma / q / mass."""
    p = {
        "log10_M200_TwoPowerTriaxial_halo": log10M200,
        "ln_cvprime_TwoPowerTriaxial_halo": np.log(12.0),
        "gamma_TwoPowerTriaxial_halo": gamma,
        "beta_TwoPowerTriaxial_halo": 3.0,
        "q_TwoPowerTriaxial_halo": 0.8,
    }
    halo = _halo_params_m200c(agama, p, _CFG)
    M200 = 10.0**log10M200
    r200 = (3.0 * M200 / (4.0 * np.pi * _CFG["halo_Delta_mass"] * _rho_crit())) ** (1.0 / 3.0)
    assert halo["scaleRadius"] > 0 and np.isfinite(halo["scaleRadius"])
    assert agama.Potential(**halo).enclosedMass(r200) == pytest.approx(M200, rel=1e-6)


@pytest.mark.parametrize("q", [1.0, 0.8, 1.2])
def test_nfw_halo_scale_radius_is_r200_over_c200(q):
    """At gamma = 1 the m200_c mapping collapses to the textbook NFW relation r_s = r200 / c200.

    This is the whole reason the NFW variant pins gamma: in general
    ``r_h = r200 / [c200 (2 - gamma)]``, so gamma rescales the halo at fixed (M200, c) instead of
    only reshaping it — the coupling behind gamma's posterior railing. At gamma = 1 the (2 - gamma)
    factor is unity, and the 94 -> 200 concentration conversion (which assumes the NFW m(c)) is
    exact rather than approximate. Independent of the flattening q.
    """
    p = {
        "log10_M200_TwoPowerTriaxial_halo": np.log10(1.30e12),
        "ln_cvprime_TwoPowerTriaxial_halo": np.log(15.4),
        "gamma_TwoPowerTriaxial_halo": 1.0,
        "alpha_TwoPowerTriaxial_halo": 1.0,
        "beta_TwoPowerTriaxial_halo": 3.0,
        "q_TwoPowerTriaxial_halo": q,
    }
    halo = _halo_params_m200c(agama, p, _CFG)
    M200 = 10.0 ** p["log10_M200_TwoPowerTriaxial_halo"]
    r200 = (3.0 * M200 / (4.0 * np.pi * _CFG["halo_Delta_mass"] * _rho_crit())) ** (1.0 / 3.0)
    c200 = convert_concentration(np.exp(p["ln_cvprime_TwoPowerTriaxial_halo"]), 94.0, 200.0)

    assert halo["scaleRadius"] == pytest.approx(r200 / c200, rel=1e-6)
    # (alpha, beta, gamma) = (1, 3, 1) is AGAMA's Spheroid form of NFW; the flattening survives.
    assert (halo["alpha"], halo["beta"], halo["gamma"]) == (1.0, 3.0, 1.0)
    assert halo["axisRatioZ"] == pytest.approx(q)
    assert agama.Potential(**halo).enclosedMass(r200) == pytest.approx(M200, rel=1e-6)


def test_nfw_simulator_config_pins_the_halo_exponents(compose):
    """The NFW config fixes gamma/alpha/beta (so they are not inferred) and keeps q/p/tilt free."""
    from hydrabflow.simulators.registry import get_simulator

    cfg = compose(overrides=["simulator=stream_agama_rnbody_ibata_m200c_nfw", "composition=global"])
    sim = get_simulator(cfg.simulator)
    globals_ = set(sim.global_parameter_names)

    for pinned in (
        "gamma_TwoPowerTriaxial_halo",
        "alpha_TwoPowerTriaxial_halo",
        "beta_TwoPowerTriaxial_halo",
        "rho_TwoPowerTriaxial_halo",
        "a_TwoPowerTriaxial_halo",
    ):
        assert pinned not in globals_, f"{pinned} should be an identity constant, not inferred"
    for free in (
        "log10_M200_TwoPowerTriaxial_halo",
        "ln_cvprime_TwoPowerTriaxial_halo",
        "q_TwoPowerTriaxial_halo",
        "p_TwoPowerTriaxial_halo",
        "tilt_TwoPowerTriaxial_halo",
    ):
        assert free in globals_, f"{free} should be inferred"

    theta = sim.sample_prior(4, rng=np.random.default_rng(0))
    assert np.allclose(theta["gamma_TwoPowerTriaxial_halo"], 1.0)
    assert np.allclose(theta["alpha_TwoPowerTriaxial_halo"], 1.0)
    assert np.allclose(theta["beta_TwoPowerTriaxial_halo"], 3.0)


def test_host_potential_dispatch_matches_equivalent_rho_a():
    """m200_c dispatch builds a potential whose halo == the rho_a halo with the derived (r_h, rho0)."""
    p = {
        "log10_M200_TwoPowerTriaxial_halo": np.log10(1.30e12),
        "ln_cvprime_TwoPowerTriaxial_halo": np.log(15.4),
        "gamma_TwoPowerTriaxial_halo": 1.0,
        "beta_TwoPowerTriaxial_halo": 3.0,
        "q_TwoPowerTriaxial_halo": 1.0,
        # disk params (single thin disk); values are unimportant, just need to be present
        "r_Disk": 2.5,
        "z_Disk": 0.3,
        "Sigma_Disk": 5.0e8,
    }
    cfg = dict(
        _CFG,
        gas_disks=False,
        thick_disk=False,
        disk_vertical="exponential",
        bulge_density_norm=9.93e10,
        halo_parameterization="m200_c",
    )
    pot = _host_potential(agama, p, cfg)
    # rebuild with the rho_a path using the derived halo params -> same force everywhere
    halo = _halo_params_m200c(agama, p, cfg)
    p_rhoa = {**p, "rho_TwoPowerTriaxial_halo": halo["densityNorm"],
              "a_TwoPowerTriaxial_halo": halo["scaleRadius"]}
    cfg_rhoa = dict(cfg, halo_parameterization="rho_a", halo_r_t_kpc=_CFG["halo_r_t_kpc"])
    pot_rhoa = _host_potential(agama, p_rhoa, cfg_rhoa)
    x = [15.0, 0.0, 3.0]
    assert np.allclose(pot.force(x), pot_rhoa.force(x), rtol=1e-6)


# ----------------------------------------------------------------------------------------------- #
# simulate() stores the (densityNorm, scaleRadius) AGAMA received per row under *_derived keys when
# the halo is parameterized by (M200, c_v') — diagnostics only, NOT inferred. Absent for rho_a.
# ----------------------------------------------------------------------------------------------- #

def _priors_local_ident():
    ident = lambda v: {"type": "identity", "prior_parameters": [v]}  # noqa: E731
    norm = lambda m, s: {"type": "normal", "prior_parameters": [m, s]}  # noqa: E731
    return dict(m_progenitor=ident(5e4), a_progenitor=ident(6.0), t_end=ident(1.5),
                ra=ident(180.0), dec=ident(-20.0), vr=norm(-58.6, 0.2), r=norm(20.6, 0.2),
                mu_ra_cosdec=norm(-2.736, 0.05), mu_dec=norm(-2.646, 0.05))


def _m200c_params():
    return dict(
        n_particles=200, n_workers=2,
        halo_parameterization="m200_c", halo_H0_kms_mpc=70.4,
        halo_Delta_mass=200.0, halo_Delta_c=94.0, halo_r_t_kpc=1000.0,
        target_streams={"Pal5": 0},
        priors_global=dict(
            log10_M200_TwoPowerTriaxial_halo={"type": "uniform", "prior_parameters": [11.7, 12.4]},
            ln_cvprime_TwoPowerTriaxial_halo={"type": "normal", "prior_parameters": [2.56, 0.272]},
            gamma_TwoPowerTriaxial_halo={"type": "uniform", "prior_parameters": [-2.0, 1.5]},
            beta_TwoPowerTriaxial_halo={"type": "identity", "prior_parameters": [3.0]},
            q_TwoPowerTriaxial_halo={"type": "uniform", "prior_parameters": [0.5, 1.5]},
            rho_TwoPowerTriaxial_halo={"type": "identity", "prior_parameters": [8.54e6]},
            a_TwoPowerTriaxial_halo={"type": "identity", "prior_parameters": [19.6]},
            r_Disk={"type": "normal", "prior_parameters": [2.6, 0.5]},
            z_Disk={"type": "normal", "prior_parameters": [0.3, 0.05]},
            Sigma_Disk={"type": "uniform", "prior_parameters": [1.0e7, 1.5e9]},
        ),
        priors_local=dict(Pal5=_priors_local_ident()),
    )


def test_simulate_stores_derived_rho_a_for_m200c():
    """m200_c: simulate() emits rho/a_..._derived (n,1) equal to the per-row _halo_params_m200c
    solve, while log10_M200/ln_cvprime are the inferred params (rho/a stay identity constants)."""
    sim = AgamaStreamSimulator(_m200c_params())
    assert "log10_M200_TwoPowerTriaxial_halo" in sim.global_parameter_names
    assert "ln_cvprime_TwoPowerTriaxial_halo" in sim.global_parameter_names
    # rho/a are identity -> NOT inferred
    assert "rho_TwoPowerTriaxial_halo" not in sim.global_parameter_names
    assert "a_TwoPowerTriaxial_halo" not in sim.global_parameter_names

    params = sim.sample_prior(3, np.random.default_rng(0))
    out = sim.simulate(params, np.random.default_rng(1))
    for key in ("rho_TwoPowerTriaxial_halo_derived", "a_TwoPowerTriaxial_halo_derived"):
        assert out[key].shape == (3, 1)
        assert np.isfinite(out[key]).all() and np.all(out[key] > 0)
    # match the direct conversion for each row
    cfg = sim._pot_cfg
    for i in range(3):
        p = {k: float(np.asarray(v).reshape(3, -1)[i, 0]) for k, v in params.items()}
        h = _halo_params_m200c(agama, p, cfg)
        assert out["rho_TwoPowerTriaxial_halo_derived"][i, 0] == pytest.approx(h["densityNorm"], rel=1e-6)
        assert out["a_TwoPowerTriaxial_halo_derived"][i, 0] == pytest.approx(h["scaleRadius"], rel=1e-6)


def test_simulate_no_derived_keys_for_rho_a():
    """Legacy rho_a halo: simulate() does NOT emit the *_derived diagnostic keys."""
    params = _m200c_params()
    params["halo_parameterization"] = "rho_a"
    params["priors_global"]["rho_TwoPowerTriaxial_halo"] = {
        "type": "uniform", "prior_parameters": [1.5e6, 0.5e8]}
    params["priors_global"]["a_TwoPowerTriaxial_halo"] = {
        "type": "uniform", "prior_parameters": [1.0, 30.0]}
    sim = AgamaStreamSimulator(params)
    out = sim.simulate(sim.sample_prior(2, np.random.default_rng(0)), np.random.default_rng(1))
    assert "rho_TwoPowerTriaxial_halo_derived" not in out
    assert "a_TwoPowerTriaxial_halo_derived" not in out


def test_rnbody_simulate_m200c_ancillary_and_derived_keys():
    """The restricted-N-body simulator goes through the same simulate() assembly (_row_jobs
    seam), so with an Ibata m200_c config it must emit the ancillary observables AND the derived
    rho/a diagnostics, with the derived values matching the per-row _halo_params_m200c solve."""
    from hydrabflow.simulators.stream_agama_rnbody import RestrictedNbodyStreamSimulator

    params = _m200c_params()
    params.update(
        n_particles=60, n_workers=2,
        # tiny restricted-N-body settings so the test stays fast
        n_updates=3, traj_per_update=8, agama_num_threads=1,
        # Ibata potential + observables (as in stream_agama_rnbody_ibata_onedisk_beta3_m200c)
        gas_disks=True, thick_disk=False, disk_vertical="exponential",
        bulge_density_norm=9.93e10,
        ancillary_observables=["vterm", "sigma_z", "rho_z"],
    )
    params["priors_local"]["Pal5"]["t_end"] = {"type": "identity", "prior_parameters": [0.3]}
    sim = RestrictedNbodyStreamSimulator(params)

    prior = sim.sample_prior(2, np.random.default_rng(0))
    out = sim.simulate(prior, np.random.default_rng(1))

    assert out["sim_data_projected"].shape == (2, 60, 6)
    spec = sim._ancillary_spec
    assert out["vterm_kms"].shape == (2, spec["l_deg"].size, 1)
    assert out["sigma_z"].shape == (2, 1)
    assert out["rho_z"].shape == (2, spec["z_kpc"].size, 1)
    for key in ("vterm_kms", "sigma_z", "rho_z"):
        assert np.isfinite(out[key]).all()

    cfg = sim._pot_cfg
    for key in ("rho_TwoPowerTriaxial_halo_derived", "a_TwoPowerTriaxial_halo_derived"):
        assert out[key].shape == (2, 1)
        assert np.isfinite(out[key]).all() and np.all(out[key] > 0)
    for i in range(2):
        p = {k: float(np.asarray(v).reshape(2, -1)[i, 0]) for k, v in prior.items()}
        h = _halo_params_m200c(agama, p, cfg)
        assert out["rho_TwoPowerTriaxial_halo_derived"][i, 0] == pytest.approx(
            h["densityNorm"], rel=1e-6)
        assert out["a_TwoPowerTriaxial_halo_derived"][i, 0] == pytest.approx(
            h["scaleRadius"], rel=1e-6)


# ------------------------------------------------------------------------------------------- #
# Present-day bound remnant mass (restricted N-body only)
# ------------------------------------------------------------------------------------------- #


def test_bound_mass_recovers_a_planted_remnant():
    """``_bound_mass`` must find a bound Plummer remnant embedded in an unbound debris cloud, and
    must locate it even though the supplied centre is offset (our centre trajectory is the massless
    test-particle orbit, which drifts from the real remnant over a few Gyr)."""
    import agama

    from hydrabflow.simulators.stream_agama_rnbody import _bound_mass, _plummer_sample

    agama.setUnits(mass=1, length=1, velocity=1)
    rng = np.random.default_rng(0)
    n_bound, n_debris = 300, 700
    m_total = 1.0e5
    scale = 0.005  # 5 pc
    per = m_total / (n_bound + n_debris)

    pos, vel = _plummer_sample(agama, n_bound, m_total * n_bound / (n_bound + n_debris), scale, rng)
    centre = np.array([8.0, 1.0, 0.5, 10.0, 200.0, -5.0])
    remnant = np.hstack([pos + centre[0:3], vel + centre[3:6]])
    # debris: spread over kpc with a large velocity offset -> unambiguously unbound
    debris = np.hstack(
        [
            centre[0:3] + rng.normal(0.0, 1.5, (n_debris, 3)),
            centre[3:6] + rng.normal(0.0, 40.0, (n_debris, 3)),
        ]
    )
    xv = np.vstack([remnant, debris])
    masses = np.full(len(xv), per)

    offset = centre + np.array([0.3, -0.2, 0.1, 3.0, -4.0, 2.0])
    m_bound = _bound_mass(agama, xv, masses, offset, scale)

    planted = per * n_bound
    assert 0.3 * planted < m_bound < 1.4 * planted
    assert m_bound < m_total


def test_bound_mass_returns_zero_for_a_fully_dissolved_cluster():
    """No overdensity anywhere -> nothing bound. (This is what a progenitor whose *present-day* mass
    is used as its *initial* mass does over a few Gyr, which is why the diagnostic is worth saving.)"""
    import agama

    from hydrabflow.simulators.stream_agama_rnbody import _bound_mass

    agama.setUnits(mass=1, length=1, velocity=1)
    rng = np.random.default_rng(1)
    n = 500
    xv = np.hstack([rng.normal(0.0, 2.0, (n, 3)), rng.normal(0.0, 50.0, (n, 3))])
    xv[:, 0] += 8.0
    masses = np.full(n, 4.3e3 / n)  # Pal 5-like: a ~2 km/s well against a 50 km/s spread
    assert _bound_mass(agama, xv, masses, np.array([8.0, 0, 0, 0, 0, 0.0]), 0.00843) == 0.0


# ------------------------------------------------------------------------------------------- #
# Solar phase-space frame (marginalized nuisance)
# ------------------------------------------------------------------------------------------- #


def _rho_a_pot_cfg():
    return dict(
        halo_r_t_kpc=1000.0, gas_disks=True, thick_disk=False, disk_vertical="exponential",
        bulge_density_norm=9.93e10, halo_parameterization="rho_a",
        halo_H0_kms_mpc=70.4, halo_Delta_mass=200.0, halo_Delta_c=94.0,
    )


def _rho_a_params(**extra):
    p = dict(
        rho_TwoPowerTriaxial_halo=8.5e6, a_TwoPowerTriaxial_halo=19.6,
        gamma_TwoPowerTriaxial_halo=1.0, beta_TwoPowerTriaxial_halo=3.0,
        q_TwoPowerTriaxial_halo=0.9, r_Disk=2.6, z_Disk=0.3, Sigma_Disk=8.0e8,
    )
    p.update(extra)
    return p


def test_solar_frame_defaults_and_peculiar_v():
    """Without priors the frame is the fixed literature default; V_Sun is PECULIAR, so the frame's
    azimuthal velocity is v_circ(R0) + V_Sun in this row's own potential."""
    import agama

    from hydrabflow.simulators.stream_agama import Z_SUN_KPC, _host_potential, _solar_frame
    from hydrabflow.simulators.stream_common import R0_KPC, vcirc_from_potential

    agama.setUnits(mass=1, length=1, velocity=1)
    cfg, p = _rho_a_pot_cfg(), _rho_a_params()
    pot = _host_potential(agama, p, cfg)

    R0, vx, vy, vz, z_sun = _solar_frame(agama, pot, p)
    assert R0 == pytest.approx(R0_KPC)
    assert (vx, vz, z_sun) == pytest.approx((11.1, 7.25, Z_SUN_KPC))
    v_circ = float(vcirc_from_potential(pot, R0_KPC)[0])
    assert vy == pytest.approx(v_circ + 12.24)

    # a drawn frame is used verbatim, and V stays peculiar w.r.t. the NEW R0
    q = _rho_a_params(R0_Sun=8.05, U_Sun=13.0, V_Sun=10.0, W_Sun=6.5)
    R0b, vxb, vyb, vzb, _ = _solar_frame(agama, pot, q)
    assert (R0b, vxb, vzb) == pytest.approx((8.05, 13.0, 6.5))
    assert vyb == pytest.approx(float(vcirc_from_potential(pot, 8.05)[0]) + 10.0)


def test_solar_frame_projection_round_trip_is_exact():
    """The projection and the progenitor's ICRS -> Galactocentric conversion must use ONE frame:
    pushing observed coordinates through both directions has to return them unchanged. This is the
    guard against the two desynchronising when R0/v_sun are varied."""
    import agama

    from hydrabflow.simulators.stream_agama import _host_potential, _solar_frame
    from hydrabflow.simulators.stream_common import _sky_projection_agama

    agama.setUnits(mass=1, length=1, velocity=1)
    p = _rho_a_params(R0_Sun=8.05, U_Sun=13.0, V_Sun=10.0, W_Sun=6.5)
    pot = _host_potential(agama, p, _rho_a_pot_cfg())
    frame = _solar_frame(agama, pot, p)

    obs = (229.022, -0.112, 20.6, -2.736, -2.646, -58.6)  # Pal 5
    ra, dec, dist, pmra, pmdec, vlos = obs
    lon, lat, pmlon, pmlat = agama.transformCelestialCoords(
        agama.fromICRStoGalactic, np.radians(ra), np.radians(dec), pmra, pmdec
    )
    xv = np.array(
        agama.getGalactocentricFromGalactic(
            lon, lat, dist, pmlon * 4.74, pmlat * 4.74, vlos,
            galcen_distance=frame[0], galcen_v_sun=frame[1:4], z_sun=frame[4],
        )
    )
    back = _sky_projection_agama(xv[None, None, :], np.array([frame]))[0, 0]
    np.testing.assert_allclose(back, np.asarray(obs), rtol=0, atol=1e-9)


def test_sky_projection_agama_matches_astropy_on_the_default_frame():
    """The AGAMA path reproduces the astropy default-frame projection to well within Gaia errors
    (the two ICRS<->Galactic rotation matrices differ slightly, which is why astropy stays the
    default when the Solar frame is not varied)."""
    from hydrabflow.simulators.stream_common import _sky_projection_agama, sky_projection

    rng = np.random.default_rng(0)
    xv = np.concatenate(
        [rng.uniform(-25.0, 25.0, (1, 400, 3)), rng.normal(0.0, 150.0, (1, 400, 3))], axis=-1
    )
    astro = sky_projection(xv)
    # astropy's own default Galactocentric frame values
    frame = np.array([[8.122, 12.9, 245.6, 7.78, 0.0208]])
    ag = _sky_projection_agama(xv, frame)

    assert np.abs(ag[..., 0] - astro[..., 0]).max() < 5e-3     # ra  [deg]
    assert np.abs(ag[..., 1] - astro[..., 1]).max() < 5e-3     # dec [deg]
    np.testing.assert_allclose(ag[..., 2], astro[..., 2], rtol=1e-10)   # distance
    assert np.abs(ag[..., 3] - astro[..., 3]).max() < 5e-2     # pm  [mas/yr]
    assert np.abs(ag[..., 4] - astro[..., 4]).max() < 5e-2
    np.testing.assert_allclose(ag[..., 5], astro[..., 5], rtol=1e-9)    # v_los


def test_solar_params_are_marginalized_not_inferred(compose):
    """`params.marginalize` keeps the Solar parameters out of the inferred set (and therefore out of
    the compositional prior score) while still drawing them per row for the forward model."""
    from hydrabflow.simulators.registry import get_simulator

    sim = get_simulator(compose(["simulator=stream_agama_rnbody_ibata_m200c_v2"]).simulator)
    solar = ["R0_Sun", "U_Sun", "V_Sun", "W_Sun"]
    assert sim._marginalized == set(solar)
    for name in solar:
        assert name not in sim.global_parameter_names
        assert name not in sim.prior_spec_global
        assert name in sim._priors_global  # still declared, so still drawn

    drawn = sim.sample_prior(4000, np.random.default_rng(0))
    for name, (mu, sd) in zip(solar, [(8.178, 0.026), (11.1, 1.25), (12.24, 2.05), (7.25, 0.62)]):
        arr = np.asarray(drawn[name]).ravel()
        assert arr.mean() == pytest.approx(mu, abs=4.0 * sd / np.sqrt(len(arr)))
        assert arr.std() == pytest.approx(sd, rel=0.1)
