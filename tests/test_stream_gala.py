"""gala particle-spray simulator (`stream_gala`): declaration layer, the q_rho <-> c_phi map, the
(M200, c) -> NFW map, config wiring with the 2-modality stacks, and (when gala is installed) a tiny
end-to-end simulate."""

from __future__ import annotations

import numpy as np
import pytest

SIM = "simulator=stream_gala_spray_mw22"
GLOBALS = ["log10_M200_halo", "c200_halo", "q_rho_halo", "m_disk", "h_R_disk", "h_z_disk"]
LOCALS = ["m_progenitor", "t_end", "vr", "r", "mu_ra_cosdec", "mu_dec"]
R_S_MW22 = 15.626


def test_stream_gala_registered():
    from hydrabflow.registry import SIMULATORS

    SIMULATORS.discover()
    assert "stream_gala" in SIMULATORS.items


def test_declares_hierarchy(compose):
    from hydrabflow.registry import get_simulator

    cfg = compose([SIM], fill=False)
    sim = get_simulator(cfg.simulator)
    assert sim.global_parameter_names == GLOBALS
    assert sim.local_parameter_names == LOCALS
    assert sim.context_keys == ["j"]
    assert sim.observable_keys == ["sim_data_projected", "vcirc_kms"]
    assert set(sim.prior_spec_global) == set(GLOBALS)
    assert set(sim.prior_spec_local) == {"Pal5", "NGC3201", "M68"}
    assert sim._gala_opts["n_steps"] == 4999 and sim._gala_opts["integrator"] == "leapfrog"


def test_q_rho_c_phi_map_anchors():
    """Numbers from the 2026-09-20 derivation (r_s = 15.626, r_ref = 15 kpc)."""
    from hydrabflow.simulators.stream_gala import (
        c_phi_from_q_rho,
        halo_rho_negative_radius,
        q_rho_from_c_phi,
    )

    for c, q in [(0.75, 0.489), (0.90, 0.779), (1.0, 1.0), (1.10, 1.239), (1.20, 1.491)]:
        assert abs(q_rho_from_c_phi(c, R_S_MW22, 15.0) - q) < 2e-3
    assert abs(c_phi_from_q_rho(q_rho_from_c_phi(0.83, R_S_MW22, 15.0), R_S_MW22, 15.0) - 0.83) < 1e-6
    # potential flattening below ~0.888 -> negative density on the pole, but only far out
    assert np.isinf(halo_rho_negative_radius(0.95, R_S_MW22))
    assert 100.0 < halo_rho_negative_radius(0.85, R_S_MW22) < 200.0
    assert 30.0 < halo_rho_negative_radius(0.75, R_S_MW22) < 60.0


def test_flattened_density_matches_finite_difference_laplacian():
    from hydrabflow.simulators.stream_gala import flattened_nfw_density

    rs, G = R_S_MW22, 1.0

    def phi(x, y, z, c):
        u = np.sqrt(x * x + y * y + z * z / c**2) / rs
        return -G / rs * np.log1p(u) / u

    def fd(x, y, z, c, h=1e-3):
        f = lambda dx, dy, dz: phi(x + dx, y + dy, z + dz, c)  # noqa: E731
        lap = (f(h, 0, 0) + f(-h, 0, 0) + f(0, h, 0) + f(0, -h, 0) + f(0, 0, h) + f(0, 0, -h)
               - 6 * f(0, 0, 0)) / h**2
        return lap / (4 * np.pi * G)

    for x, y, z, c in [(5, 3, 2, 0.8), (10, 0, 7, 1.2), (1, 1, 20, 0.9)]:
        a = flattened_nfw_density(np.hypot(x, y), z, 1.0, rs, c, G=G)
        assert abs(a - fd(x, y, z, c)) < 1e-5 * abs(fd(x, y, z, c))


def test_c_phi_bracket_covers_prior_corners():
    """Every (M200, c200, q_rho) corner of the prior inverts inside the config bracket."""
    from hydrabflow.simulators.stream_gala import c_phi_from_q_rho, nfw_m_rs_from_m200_c

    for lm in (11.6, 12.4):
        for c200 in (8.0, 20.0):
            _, r_s = nfw_m_rs_from_m200_c(10**lm, c200)
            for q in (0.5, 1.5):
                c = c_phi_from_q_rho(q, r_s, 15.0, bracket=(0.6, 1.6))
                assert 0.7 < c < 1.25
    with pytest.raises(ValueError, match="bracket"):
        c_phi_from_q_rho(0.5, R_S_MW22, 15.0, bracket=(0.9, 1.1))


def test_m200_c_to_nfw_matches_gala_formula_and_mw22():
    from hydrabflow.simulators.stream_gala import RHO_CRIT_PLANCK18_MSUN_KPC3, nfw_m_rs_from_m200_c

    # Planck18 critical density (astropy) — gala's from_M200_c default cosmology
    import astropy.units as u
    from astropy.cosmology import Planck18

    rho_c = Planck18.critical_density(0).to(u.Msun / u.kpc**3).value
    assert abs(rho_c / RHO_CRIT_PLANCK18_MSUN_KPC3 - 1) < 1e-4
    # MilkyWayPotential2022's halo (m = 5.5427e11, r_s = 15.626) is M200 = 9.597e11, c200 = 13.32
    m, r_s = nfw_m_rs_from_m200_c(9.597e11, 13.32)
    assert abs(m / 5.5427e11 - 1) < 2e-3 and abs(r_s / 15.626 - 1) < 2e-3


def test_config_composes_with_both_2modal_stacks(compose):
    from hydrabflow.pipeline.adapter import fill_stream_grid_from_simulator

    for extra in (
        ["model=stream_fusion_2modal", "adapter=stream_2modal", "augmentation=stream_global_ibata_grid_v2",
         "preprocessing=stream_global_log10_gala_2modal", "eval=stream_compositional_masked"],
        ["model=stream_fusion_2modal_particles", "adapter=stream_2modal_particles", "augmentation=stream_global",
         "preprocessing=stream_global_log10_gala_2modal", "eval=stream_compositional_masked_particles"],
    ):
        cfg = compose([SIM, "composition=global"] + extra)
        fill_stream_grid_from_simulator(cfg)
        assert list(cfg.adapter.inference_variables) == GLOBALS
        assert len(cfg.augmentation.params.obs_r_kpc) == 19
        assert cfg.augmentation.params.obs_r_kpc[0] == 6.6 and cfg.augmentation.params.obs_r_kpc[-1] == 25.53
        assert [s.name for s in cfg.preprocessing.steps][-1] == "log10_transform"
        assert list(cfg.preprocessing.steps[-1]["keys"]) == ["m_disk", "h_R_disk", "h_z_disk"]


def test_real_preset_declares_gala_log10_keys(compose):
    cfg = compose([SIM, "preprocessing=stream_real_global_log10_gala"], fill=False)
    assert [s.name for s in cfg.preprocessing.steps] == [
        "attach_observed_vcirc", "mask_vcirc_radii", "log10_transform"
    ]
    assert list(cfg.preprocessing.steps[-1]["keys"]) == ["m_disk", "h_R_disk", "h_z_disk"]


def test_store_window_matches_augmentation_window(compose):
    from omegaconf import OmegaConf

    from hydrabflow.registry import get_simulator

    cfg = compose([SIM, "augmentation=stream_global"], fill=False)
    sim_tab = OmegaConf.to_container(cfg.simulator.params.store_window_subsample.observational_window)
    aug_tab = OmegaConf.to_container(cfg.augmentation.params.observational_window)
    assert sim_tab == aug_tab
    sub = get_simulator(cfg.simulator)._store_window_subsample
    assert sub["max_particles"] == 2000 and sub["pad_value"] == -999.0


def test_particle_count_identity_guard(compose):
    from hydrabflow.registry import get_simulator

    cfg = compose([SIM, "simulator.params.n_steps=5000"], fill=False)
    with pytest.raises(ValueError, match="n_steps"):
        _ = get_simulator(cfg.simulator)._gala_opts
    cfg = compose([SIM, "+simulator.params.vcirc_rejection={max_frac_dev: 0.2}"], fill=False)
    with pytest.raises(ValueError, match="vcirc_rejection"):
        _ = get_simulator(cfg.simulator)._gala_opts


def test_simulate_smoke_with_gala(compose):
    pytest.importorskip("gala")
    from hydrabflow.registry import get_simulator

    cfg = compose(
        [SIM, "simulator.params.n_particles=400", "simulator.params.n_steps=199",
         "simulator.params.n_workers=1", "simulator.params.store_window_subsample=null",
         "+simulator.params.raise_errors=true"],
        fill=False,
    )
    sim = get_simulator(cfg.simulator)
    out = sim.sample(2, np.random.default_rng(0))
    assert out["sim_data_projected"].shape == (2, 400, 6) and np.isfinite(out["sim_data_projected"]).all()
    assert out["vcirc_kms"].shape == (2, 19, 1) and np.isfinite(out["vcirc_kms"]).all()
    assert (out["vcirc_kms"] > 100).all() and (out["vcirc_kms"] < 400).all()
    for key in ("c_phi_halo_derived", "halo_rho_neg_r_kpc_derived", "m_nfw_halo_derived", "r_s_halo_derived"):
        assert out[key].shape == (2, 1)
    assert (out["c_phi_halo_derived"] > 0.6).all() and (out["c_phi_halo_derived"] < 1.6).all()
    comp = sim.sample_compositional(1, np.random.default_rng(1))
    assert comp["sim_data_projected"].shape == (1, 3, 400, 6)
    assert comp["vcirc_kms"].shape == (1, 19, 1) and comp["c_phi_halo_derived"].shape == (1, 1)
    assert comp["m_progenitor"].shape == (1, 3, 1)
