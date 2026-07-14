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
