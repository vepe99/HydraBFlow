"""Tests for the Frenet-Serret remap simulator (stream_trihedron).

Covers the exactness anchor (remapping into the fiducial potential returns the cloud the template
was built from), the guard that stops a config freeing-and-then-inferring the stripping history, and
that the simulator is a drop-in for the v4 stack's output contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

agama = pytest.importorskip("agama")

from hydra import compose, initialize_config_dir  # noqa: E402

from hydrabflow.config import register_configs  # noqa: E402
from hydrabflow.registry import get_simulator  # noqa: E402
from hydrabflow.simulators import trihedron as tri  # noqa: E402
from hydrabflow.simulators.stream_agama import _host_potential, _solar_frame  # noqa: E402
from hydrabflow.simulators.stream_trihedron import (  # noqa: E402
    FROZEN_LOCAL_KEYS,
    TrihedronStreamSimulator,
    load_templates,
    template_fiducial_locals,
)

@pytest.fixture(scope="module", autouse=True)
def _units():
    """The template's t_knots are in the (kpc, km/s, Msun) time unit the workers use; without this
    the parent process would integrate the check orbit in agama's default units."""
    agama.setUnits(length=1, velocity=1, mass=1)


REPO = Path(__file__).resolve().parent.parent
CONF = REPO / "conf"
SIM_NAME = "stream_trihedron_mcmillan17_v4"
FIDUCIAL = REPO / "data_local/mcmillan17_mean_tend_best/mcmillan17_mean.npz"


@pytest.fixture(scope="module")
def sim():
    register_configs()
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"simulator={SIM_NAME}", "composition=global"],
        )
    s = get_simulator(cfg.simulator)
    if not Path(s._template_path).exists():
        pytest.skip(f"trihedron template not built ({s._template_path})")
    return s


def test_registry_resolves_stream_trihedron(sim):
    assert isinstance(sim, TrihedronStreamSimulator)


def test_stripping_history_is_frozen_not_inferred(sim):
    """The remap cannot respond to m/a/t_end, so they must not reach local_parameter_names."""
    assert sim.local_parameter_names == ["vr", "r", "mu_ra_cosdec", "mu_dec"]
    assert not set(FROZEN_LOCAL_KEYS) & set(sim.local_parameter_names)
    # The potential globals are the v4 set; the halo is (M200, c'), not (rho, a).
    assert "log10_M200_TwoPowerTriaxial_halo" in sim.global_parameter_names
    assert "rho_TwoPowerTriaxial_halo" not in sim.global_parameter_names


def test_template_prior_mismatch_is_rejected(sim):
    """Freeing a frozen local must fail loudly: a network would otherwise be trained to infer a
    parameter the forward model ignores."""
    sim._check_template_matches_priors()  # the shipped config is consistent

    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"simulator={SIM_NAME}",
                "composition=global",
                "simulator.params.priors_local.Pal5.t_end.type=uniform",
                "simulator.params.priors_local.Pal5.t_end.prior_parameters=[2.0,5.0]",
            ],
        )
    bad = get_simulator(cfg.simulator)
    with pytest.raises(ValueError, match="t_end"):
        bad._check_template_matches_priors()


def test_remap_is_exact_at_the_fiducial_potential(sim):
    """The template's one exactness anchor, and the check that its labels belong to the right orbit."""
    if not FIDUCIAL.exists():
        pytest.skip("fiducial simulation not present")
    data = np.load(FIDUCIAL)
    xv_all = np.asarray(data["sim_data_carthesian"], dtype=float)[0]
    templates = load_templates(sim._template_path)

    # Rebuild the fiducial (McMillan17) potential from the stored globals, as the builder did.
    row_g = {
        k: float(np.asarray(data[k])[0, 0])
        for k in data.files
        if np.asarray(data[k]).ndim == 2 and np.asarray(data[k]).shape[1] == 1
    }
    pot_cfg = dict(
        halo_r_t_kpc=float("inf"), gas_disks=True, thick_disk=True,
        disk_vertical="isothermal", bulge_density_norm=9.8351e10,
        halo_parameterization="rho_a",
    )
    for s, j in enumerate(np.asarray(data["j"]).reshape(-1).astype(int)):
        row = dict(row_g)
        row.update({
            k: float(np.asarray(data[k])[0, s, 0])
            for k in data.files
            if np.asarray(data[k]).ndim == 3 and np.asarray(data[k]).shape[2] == 1
        })
        pot = _host_potential(agama, row, pot_cfg)
        frame = _solar_frame(agama, pot, row)
        l0, b0, pml0, pmb0 = agama.transformCelestialCoords(
            agama.fromICRStoGalactic,
            row["ra"] * np.pi / 180, row["dec"] * np.pi / 180,
            row["mu_ra_cosdec"], row["mu_dec"],
        )
        posvel = np.array(agama.getGalactocentricFromGalactic(
            l0, b0, row["r"], pml0 * 4.74, pmb0 * 4.74, row["vr"],
            galcen_distance=frame[0], galcen_v_sun=frame[1:4], z_sun=frame[4],
        ))
        back = tri.remap(agama, pot, posvel, templates[int(j)])
        assert np.nanmax(np.abs(back[:, 0:3] - xv_all[s][:, 0:3])) < 1e-6


def test_template_records_the_frozen_locals(sim):
    fiducial = template_fiducial_locals(sim._template_path)
    assert set(fiducial) == set(sim.target_streams.values())
    for stream, j in sim.target_streams.items():
        for key in FROZEN_LOCAL_KEYS:
            spec = sim._priors_local[stream][key]
            assert str(spec["type"]) == "identity"
            assert np.isclose(float(spec["prior_parameters"][0]), fiducial[j][key])


def test_simulate_matches_the_v4_output_contract(sim):
    rng = np.random.default_rng(0)
    params = sim.sample_prior(4, rng)
    out = sim.simulate(params, rng)

    assert out["sim_data_projected"].shape == (4, 2000, 6)          # store_window_subsample cap
    assert out["sim_data_projected"].dtype == np.float32
    assert out["vcirc_kms"].shape == (4, 19, 1)                      # Ou+2024 grid
    assert np.isfinite(out["vcirc_kms"]).all()
    # m200_c diagnostics are stored; the bound mass is not (the remap replays a fixed remnant).
    assert "rho_TwoPowerTriaxial_halo_derived" in out
    assert "a_TwoPowerTriaxial_halo_derived" in out
    assert "m_bound_final" not in out


def test_sample_compositional_shapes(sim):
    rng = np.random.default_rng(1)
    out = sim.sample_compositional(2, rng)
    m = len(sim.target_streams)
    assert out["j"].shape == (2, m, 1)
    assert out["vr"].shape == (2, m, 1)
    assert out["log10_M200_TwoPowerTriaxial_halo"].shape == (2, 1)   # one potential per group
    assert out["sim_data_projected"].shape == (2, m, 2000, 6)
    assert out["vcirc_kms"].shape == (2, 19, 1)                      # group-level observable


def test_remap_moves_the_stream_when_the_potential_changes(sim):
    """A surrogate that ignored its potential would silently produce one stream forever."""
    rng = np.random.default_rng(2)
    params = sim.sample_prior(1, rng)
    light = {k: v.copy() for k, v in params.items()}
    heavy = {k: v.copy() for k, v in params.items()}
    light["log10_M200_TwoPowerTriaxial_halo"][:] = 11.7
    heavy["log10_M200_TwoPowerTriaxial_halo"][:] = 12.3
    a = sim.simulate(light, rng)["sim_data_projected"]
    b = sim.simulate(heavy, rng)["sim_data_projected"]
    ok = (a[:, :, 0] != -999) & (b[:, :, 0] != -999)
    assert ok.sum() > 100
    assert np.nanmax(np.abs(a[:, :, 1][ok] - b[:, :, 1][ok])) > 1e-3   # dec moved
