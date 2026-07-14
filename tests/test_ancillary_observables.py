"""Sanity checks for the Ibata (2023) ancillary potential observables (new_constrains.md).

Analytic limits (flat rotation curve, Kz->Sigma conversion) + a McMillan (2017) cross-check that
validates the G constant and unit system. All use agama directly (the helpers are pure functions
of an agama Potential), so they double as a regression guard on the physics.
"""
from __future__ import annotations

import numpy as np
import pytest

agama = pytest.importorskip("agama")

from hydrabflow.simulators.stream_common import (  # noqa: E402
    G_KPC_KMS2_MSUN,
    R0_KPC,
    SIGMA_Z_OBS_MSUN_PC2,
    surface_density,
    terminal_velocity,
    vcirc_from_potential,
    vertical_density_profile,
)


@pytest.fixture(scope="module", autouse=True)
def _units():
    agama.setUnits(mass=1, length=1, velocity=1)


def test_G_matches_agama():
    # The helpers hardcode G; it must equal agama's internal G to ~1e-9 or Sigma/vterm are wrong.
    assert abs(G_KPC_KMS2_MSUN / agama.G - 1.0) < 1e-8


def test_flat_rotation_curve_and_terminal_velocity():
    # Logarithmic potential -> vc(R) = v0 = const. Then v_term(l) = sgn(sin l) v0 - v0 sin l,
    # so v_term(30 deg) = v0 (1 - 0.5) = 0.5 v0 and v_term(90 deg) = 0.
    V = 220.0
    pot = agama.Potential(type="Logarithmic", v0=V, scaleRadius=1e-6)
    assert vcirc_from_potential(pot, R0_KPC)[0] == pytest.approx(V, rel=1e-6)
    assert terminal_velocity(pot, 30.0)[0] == pytest.approx(0.5 * V, rel=1e-4)
    assert terminal_velocity(pot, -30.0)[0] == pytest.approx(-0.5 * V, rel=1e-4)
    assert terminal_velocity(pot, 90.0)[0] == pytest.approx(0.0, abs=1e-3)


def test_surface_density_kz_conversion():
    # Sigma(z) = |Kz| / (2 pi G). Build a constant-Kz slab via a Logarithmic potential's Fz and
    # check the helper inverts the conversion exactly.
    pot = agama.Potential(type="Logarithmic", v0=180.0, scaleRadius=0.5)
    Fz = float(np.asarray(pot.force([R0_KPC, 0.0, 1.1])).reshape(-1)[2])
    expected = (-Fz) / (2.0 * np.pi * G_KPC_KMS2_MSUN) / 1.0e6
    assert surface_density(pot, z=1.1) == pytest.approx(expected, rel=1e-10)


def _mcmillan_potential():
    """McMillan (2017) fallback model from new_constrains.md (halo as NFW-equivalent Spheroid)."""
    bulge = agama.Potential(type="Spheroid", densityNorm=9.93e10, axisRatioZ=0.5, gamma=0.0,
                            beta=1.8, alpha=1.0, scaleRadius=0.075, outerCutoffRadius=2.1,
                            cutoffStrength=2.0)
    gas_HI = agama.Potential(type="Disk", surfaceDensity=5.31e7, scaleRadius=7.0,
                             innerCutoffRadius=4.0, scaleHeight=0.085)
    gas_H2 = agama.Potential(type="Disk", surfaceDensity=2.18e9, scaleRadius=1.5,
                             innerCutoffRadius=12.0, scaleHeight=0.045)
    thin = agama.Potential(type="Disk", surfaceDensity=8.96e8, scaleRadius=2.5, scaleHeight=-0.30)
    thick = agama.Potential(type="Disk", surfaceDensity=1.83e8, scaleRadius=3.02, scaleHeight=-0.90)
    halo = agama.Potential(type="Spheroid", densityNorm=8.54e6, scaleRadius=19.6, gamma=1, alpha=1,
                           beta=3, outerCutoffRadius=1000.0, cutoffStrength=2)
    return agama.Potential(bulge, gas_HI, gas_H2, thin, thick, halo), agama.Potential(thin, thick)


def test_mcmillan_surface_density_and_vc():
    # Cross-check: Sigma(1.1) ~ 71 Msun/pc^2 and vc(R0) ~ 230-240 km/s for the McMillan model.
    # A wrong G / unit system / disk-height sign would move Sigma off 71 by a constant factor.
    pot, _ = _mcmillan_potential()
    assert surface_density(pot) == pytest.approx(SIGMA_Z_OBS_MSUN_PC2, abs=6.0)  # 71 +/- ~6
    assert 225.0 < vcirc_from_potential(pot, R0_KPC)[0] < 245.0


def test_vertical_density_profile_decreasing():
    _, disk = _mcmillan_potential()
    z = np.linspace(0.1, 5.0, 20)
    rho = vertical_density_profile(disk, z)
    assert rho.shape == (20,)
    assert np.all(rho > 0)
    assert np.all(np.diff(rho) < 0)  # monotonically decreasing away from the plane


# ------------------------------------------------------------------------------------------- #
# Simulator integration: the Ibata config computes/stores the ancillary observables; the legacy
# config (defaults) does not, and its potential is unchanged.
# ------------------------------------------------------------------------------------------- #

from hydrabflow.simulators.stream_agama import AgamaStreamSimulator, BULGE_PARAMS  # noqa: E402
from hydrabflow.simulators.stream_common import RHO_Z_KPC, VTERM_L_DEG  # noqa: E402


def _priors_local(vr, r, mua, mud, t_end):
    ident = lambda v: {"type": "identity", "prior_parameters": [v]}  # noqa: E731
    norm = lambda m, s: {"type": "normal", "prior_parameters": [m, s]}  # noqa: E731
    return dict(m_progenitor=ident(5e4), a_progenitor=ident(6.0), t_end=ident(t_end),
                ra=ident(180.0), dec=ident(-20.0), vr=norm(vr, 0.2), r=norm(r, 0.2),
                mu_ra_cosdec=norm(mua, 0.05), mu_dec=norm(mud, 0.05))


def _base_params():
    return dict(
        n_particles=100, n_workers=2,
        target_streams={"Pal5": 0, "NGC3201": 1, "M68": 2},
        priors_global=dict(
            rho_TwoPowerTriaxial_halo={"type": "uniform", "prior_parameters": [1.0e6, 1.5e8]},
            gamma_TwoPowerTriaxial_halo={"type": "uniform", "prior_parameters": [-2.0, 2.0]},
            a_TwoPowerTriaxial_halo={"type": "uniform", "prior_parameters": [1.0, 30.0]},
            beta_TwoPowerTriaxial_halo={"type": "identity", "prior_parameters": [3.0]},
            q_TwoPowerTriaxial_halo={"type": "uniform", "prior_parameters": [0.5, 1.5]},
            r_Disk={"type": "normal", "prior_parameters": [2.6, 0.5]},
            z_Disk={"type": "normal", "prior_parameters": [0.3, 0.05]},
            Sigma_Disk={"type": "uniform", "prior_parameters": [1.0e7, 1.5e9]},
        ),
        priors_local=dict(
            Pal5=_priors_local(-58.6, 20.6, -2.736, -2.646, 4.0),
            NGC3201=_priors_local(494.34, 4.9, 8.324, -1.991, 1.5),
            M68=_priors_local(-92.99, 10.3, -2.752, 1.762, 1.5),
        ),
    )


def test_legacy_config_has_no_ancillary_observables():
    sim = AgamaStreamSimulator(_base_params())
    assert sim.ancillary_observable_keys == []
    assert sim._pot_cfg == dict(
        halo_r_t_kpc=float("inf"), gas_disks=False, thick_disk=False, disk_vertical="isothermal",
        bulge_density_norm=BULGE_PARAMS["densityNorm"],
        halo_parameterization="rho_a", halo_H0_kms_mpc=70.4, halo_Delta_mass=200.0,
        halo_Delta_c=94.0,
    )
    out = sim.simulate(sim.sample_prior(2, np.random.default_rng(0)), np.random.default_rng(1))
    assert set(out) == {"sim_data_carthesian", "sim_data_projected", "vcirc_kms"}


def test_ibata_config_computes_ancillary_observables():
    params = _base_params()
    params.update(
        halo_r_t_kpc=1000.0, gas_disks=True, thick_disk=True, disk_vertical="exponential",
        ancillary_observables=["vterm", "sigma_z", "rho_z"],
    )
    params["priors_global"].update(
        r_thick_Disk={"type": "uniform", "prior_parameters": [1.0, 10.0]},
        dz_thick_Disk={"type": "uniform", "prior_parameters": [0.05, 4.5]},
        Sigma_thick_Disk={"type": "uniform", "prior_parameters": [1.0e7, 1.0e9]},
    )
    sim = AgamaStreamSimulator(params)
    assert sim.ancillary_observable_keys == ["vterm_kms", "sigma_z", "rho_z"]
    assert {"r_thick_Disk", "dz_thick_Disk", "Sigma_thick_Disk"} <= set(sim.global_parameter_names)

    out = sim.simulate(sim.sample_prior(3, np.random.default_rng(0)), np.random.default_rng(1))
    assert out["vterm_kms"].shape == (3, len(VTERM_L_DEG), 1)
    assert out["sigma_z"].shape == (3, 1)
    assert out["rho_z"].shape == (3, len(RHO_Z_KPC), 1)
    for k in ("vterm_kms", "sigma_z", "rho_z"):
        assert np.isfinite(out[k]).all()
    # sigma_z should be a physically plausible surface density (tens of Msun/pc^2)
    assert np.all(out["sigma_z"] > 0) and np.all(out["sigma_z"] < 500)
