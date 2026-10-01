"""rc38 host potential: disk vertical profiles, Moster-driven thin + thick disks, rc37 unchanged."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

agama = pytest.importorskip("agama")

sys.path.insert(0, str(Path(__file__).parent))
from conftest import compose_cfg  # noqa: E402

from hydrabflow.registry import get_simulator  # noqa: E402
from hydrabflow.simulators import stream_agama as sa  # noqa: E402
from hydrabflow.simulators.stream_common import (  # noqa: E402
    MOSTER13, mu_moster, sample_stream_prior, sample_stream_prior_shared_global,
)

RC37 = "stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc37"
RC38 = "stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38"
U = np.array([0.25, 0.5, 1.0, 2.0])  # heights in units of each disk's own |scaleHeight| (exp vs sech^2 differ 2x at 1)


def _sim(name):
    return get_simulator(compose_cfg([f"simulator={name}"], fill=False).simulator)


def _row(sim, seed=0):
    d = sim.sample_prior(1, np.random.default_rng(seed))
    return {k: float(v[0, 0]) for k, v in d.items()}


def _vertical_ratio(comp, R=8.0):
    """rho(R, z)/rho(R, 0) at z = U |h|. (Far above ~3 |h| AGAMA's Disk density, recovered from its
    ansatz + multipole residual, is numerically noisy and can even go slightly negative.)"""
    pot = sa._agama().Potential(**comp)
    z = U * abs(comp["scaleHeight"])
    pts = np.column_stack([np.full(z.size, R), np.zeros(z.size), z])
    return pot.density(pts) / pot.density(np.array([[R, 0.0, 0.0]]))[0]


def _exp(h):
    return np.exp(-U)  # exp(-|z|/h) at z = U h


def _sech2(h):
    return np.cosh(U / 2) ** -2  # sech^2(z/2|h|) at z = U |h|


def test_agama_scaleheight_convention():
    """AGAMA potential_disk.cpp: h > 0 exponential, h < 0 isothermal sech^2."""
    base = dict(type="Disk", surfaceDensity=1e9, scaleRadius=2.5, sersicIndex=1)
    np.testing.assert_allclose(_vertical_ratio({**base, "scaleHeight": 0.3}), _exp(0.3), rtol=0.02)
    np.testing.assert_allclose(_vertical_ratio({**base, "scaleHeight": -0.3}), _sech2(0.3), rtol=0.02)


def test_rc38_every_disk_component_has_the_intended_vertical_profile():
    sim = _sim(RC38)
    cfg = sa._resolve_pot_cfg(sim._pot_cfg)
    p = _row(sim)
    thin, thick = sa._disk_params_moster(sa._agama(), p, cfg)[0]
    np.testing.assert_allclose(_vertical_ratio(thin), _exp(0.3), rtol=0.02)     # stellar: exponential
    np.testing.assert_allclose(_vertical_ratio(thick), _exp(0.9), rtol=0.02)
    hi, h2 = sa._gas_disks(cfg)
    # gas: sech^2 (the HI/H2 disks have a central hole, so compare at R well outside it)
    np.testing.assert_allclose(_vertical_ratio(hi, R=15.0), _sech2(0.085), rtol=0.02)
    np.testing.assert_allclose(_vertical_ratio(h2, R=15.0), _sech2(0.045), rtol=0.02)


def test_rc37_components_are_unchanged():
    """rc37 keeps the legacy signs byte-for-byte: stellar -z_Disk (sech^2), gas +h (exponential)."""
    sim = _sim(RC37)
    cfg = sa._resolve_pot_cfg(sim._pot_cfg)
    assert cfg["disk_model"] == "legacy" and cfg["agama_vertical_sign"] == "legacy"
    p = _row(sim)
    disks = sa._stellar_disks(sa._agama(), p, cfg, sa._stellar_hsign(cfg))
    assert disks == [sa._thin_disk_params(p, -1.0)]
    assert disks[0]["scaleHeight"] == -p["z_Disk"]
    assert sa._gas_disks(cfg) == [sa.GAS_HI_PARAMS, sa.GAS_H2_PARAMS]
    assert sa._m200c_derived(sa._agama(), p, cfg, pot_host=object()).keys() == {
        "rho_TwoPowerTriaxial_halo_derived", "a_TwoPowerTriaxial_halo_derived"}


def test_bulge_is_mcmillan_and_its_mass_is_computed():
    b = sa.BULGE_PARAMS
    assert (b["gamma"], b["beta"], b["alpha"], b["scaleRadius"], b["outerCutoffRadius"],
            b["cutoffStrength"], b["axisRatioZ"]) == (0, 1.8, 1, 0.075, 2.1, 2, 0.5)
    m = sa._bulge_mass(sa._agama(), 9.93e10)
    assert m == pytest.approx(8.96e9, rel=5e-3)


def test_thick_thin_ratio_round_trips_mcmillan_best_fit():
    R0, z_thin, z_thick, zs = 8.21, 0.3, 0.9, sa.Z_SUN_KPC
    R_thin, S_thin, R_thick, S_thick = 2.50, 8.96e8, 3.02, 1.83e8
    rho = lambda S, z, R: S / (2 * z) * math.exp(-zs / z - R0 / R)  # noqa: E731
    f = rho(S_thick, z_thick, R_thick) / rho(S_thin, z_thin, R_thin)
    M_disk = 2 * math.pi * (S_thin * R_thin**2 + S_thick * R_thick**2)
    k = sa.thick_thin_ratio(f, R0, R_thin, R_thick, z_thin, z_thick)
    S_thin_back = M_disk / (2 * math.pi * (R_thin**2 + k * R_thick**2))
    assert S_thin_back == pytest.approx(S_thin, rel=1e-12)
    assert k * S_thin_back == pytest.approx(S_thick, rel=1e-12)
    assert f == pytest.approx(0.12, abs=0.01)  # the prior centre ln 0.12 is McMillan's value


def test_disk_derivation_masses_floor_and_r0_dependence():
    ag = sa._agama()
    cfg = sa._resolve_pot_cfg(_sim(RC38)._pot_cfg)
    p = dict(log10_Mstar=math.log10(5e10), ln_R_d_thin=math.log(2.5), ln_R_d_thick=math.log(3.0),
             ln_f_thick=math.log(0.12), R0_Sun=8.2, rho_Bulge=9.93e10)
    (thin, thick), d = sa._disk_params_moster(ag, p, cfg)
    assert d["M_disk_derived"] == pytest.approx(5e10 - d["M_bulge_derived"], rel=1e-12)
    for comp, key in ((thin, "M_thin_derived"), (thick, "M_thick_derived")):
        assert ag.Potential(**comp).totalMass() == pytest.approx(d[key], rel=2e-3)
    assert d["M_thin_derived"] + d["M_thick_derived"] == pytest.approx(d["M_disk_derived"], rel=1e-12)
    assert not d["disk_mass_floored_derived"]
    low = sa._disk_params_moster(ag, {**p, "log10_Mstar": 9.9}, cfg)[1]  # 7.9e9 < M_bulge
    assert low["disk_mass_floored_derived"] and low["M_disk_derived"] == pytest.approx(1e9)
    far = sa._disk_params_moster(ag, {**p, "R0_Sun": 8.4}, cfg)[1]  # f_thick is defined at R0
    assert far["Sigma_thin_derived"] != d["Sigma_thin_derived"]
    assert far["M_disk_derived"] == d["M_disk_derived"]


def test_rc38_config_declares_exactly_the_eight_inferred_globals():
    sim = _sim(RC38)
    assert sim.global_parameter_names == [
        "gamma_TwoPowerTriaxial_halo", "q_TwoPowerTriaxial_halo", "log10_M200_TwoPowerTriaxial_halo",
        "ln_cvprime_TwoPowerTriaxial_halo", "log10_Mstar", "ln_R_d_thin", "ln_R_d_thick", "ln_f_thick"]
    for gone in ("Sigma_Disk", "r_Disk", "z_Disk"):
        assert gone not in sim._priors_global
    assert sim.params["vcirc_rejection"] is None
    assert set(sim.params["marginalize"]) == {"R0_Sun", "U_Sun", "V_Sun", "W_Sun"}


def test_rc38_copies_everything_else_from_rc37():
    a, b = _sim(RC37), _sim(RC38)
    for key in ("priors_local", "target_streams", "store_window_subsample", "obs_r_kpc",
                "obs_vc_kms", "obs_sigma_vc", "obs_r_grid", "spray_method", "mass_loss",
                "ancillary_observables", "halo_parameterization", "gas_disks", "halo_r_t_kpc"):
        assert a._as_dict({key: a.params[key]}) == b._as_dict({key: b.params[key]}), key
    ga, gb = a._priors_global, b._priors_global
    for key in ("R0_Sun", "U_Sun", "V_Sun", "W_Sun", "rho_TwoPowerTriaxial_halo",
                "a_TwoPowerTriaxial_halo", "beta_TwoPowerTriaxial_halo", "alpha_TwoPowerTriaxial_halo",
                "p_TwoPowerTriaxial_halo", "tilt_TwoPowerTriaxial_halo", "rho_Bulge", "ln_cvprime_TwoPowerTriaxial_halo",
                "gamma_TwoPowerTriaxial_halo", "q_TwoPowerTriaxial_halo"):
        assert a._as_dict(ga[key]) == b._as_dict(gb[key]), key


@pytest.mark.parametrize("shared", [False, True])
def test_moster_conditional_is_honoured_by_both_samplers(shared):
    sim = _sim(RC38)
    draw = sample_stream_prior_shared_global if shared else sample_stream_prior
    d = draw(sim._priors_global, sim._priors_local, sim.target_streams, 200_000,
             np.random.default_rng(1))
    lm, ls = d["log10_M200_TwoPowerTriaxial_halo"][:, 0], d["log10_Mstar"][:, 0]
    resid = ls - mu_moster(lm, **MOSTER13)
    assert resid.mean() == pytest.approx(0.0, abs=2e-3)
    assert resid.std() == pytest.approx(0.2, rel=5e-3)
    assert abs(np.corrcoef(resid, lm)[0, 1]) < 5e-3
    again = draw(sim._priors_global, sim._priors_local, sim.target_streams, 200_000,
                 np.random.default_rng(1))
    np.testing.assert_array_equal(again["log10_Mstar"], d["log10_Mstar"])  # stable draw order


def test_conditional_must_follow_its_parent():
    from hydrabflow.simulators.stream_common import sample_prior_value

    spec = {"type": "moster_conditional", "prior_parameters": [0.2], "condition_on": "x"}
    with pytest.raises(ValueError, match="has not been drawn yet"):
        sample_prior_value(spec, 3, np.random.default_rng(0), drawn={})


def test_rc38_simulate_stores_bool_floor_and_derived_columns():
    sim = _sim(RC38)
    sim.params["n_particles"] = 200
    sim.params["n_workers"] = 1
    rng = np.random.default_rng(3)
    params = sim.sample_prior(2, rng)
    params["log10_Mstar"][0, 0] = 9.5  # forces the floor
    out = sim.simulate(params, rng)
    assert out["disk_mass_floored_derived"].dtype == bool
    assert out["disk_mass_floored_derived"][0, 0]
    for k in ("M50_derived", "vc_R0_derived", "Sigma_thin_derived", "M_bulge_derived"):
        assert out[k].shape == (2, 1) and np.isfinite(out[k]).all()
