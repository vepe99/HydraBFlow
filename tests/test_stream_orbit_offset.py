"""Tests for the orbit + DeltaTheta(phi1) spline simulator (stream_orbit_offset).

Covers the three claims the method rests on -- that a degree-4 spline with no interior knots IS
Ibata's quartic, that the stream-frame projection is the one the rest of the project uses and is
exactly invertible, and that the observables the model generates round-trip through the pipeline's
own projection -- plus the guard against freeing a local the forward model ignores, the v4 output
contract, and a regression test for the spline blow-up found while building this.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

agama = pytest.importorskip("agama")

from hydra import compose, initialize_config_dir  # noqa: E402

from hydrabflow.config import register_configs  # noqa: E402
from hydrabflow.registry import get_simulator  # noqa: E402
from hydrabflow.simulators import orbit_offset as oo  # noqa: E402
from hydrabflow.simulators import stream_frame as sf  # noqa: E402
from hydrabflow.simulators.stream_common import sky_projection  # noqa: E402
from hydrabflow.simulators.stream_orbit_offset import (  # noqa: E402
    OrbitOffsetStreamSimulator,
    template_fiducial_locals,
)


@pytest.fixture(scope="module", autouse=True)
def _units():
    """The template's T0/knot_dt are in the (kpc, km/s, Msun) time unit the workers use; without
    this the parent process would integrate the check orbit in agama's default units and every
    orbit-based assertion below would be quietly wrong (not merely imprecise)."""
    agama.setUnits(length=1, velocity=1, mass=1)


REPO = Path(__file__).resolve().parent.parent
CONF = REPO / "conf"
SIM_NAME = "stream_orbit_offset_mcmillan17_v4"


@pytest.fixture(scope="module")
def sim():
    register_configs()
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"simulator={SIM_NAME}", "composition=global", "++simulator.params.n_workers=2"],
        )
    s = get_simulator(cfg.simulator)
    if not Path(s._template_path).exists():
        pytest.skip(f"orbit-offset template not built ({s._template_path})")
    return s


def _random_frame(seed=0):
    R = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return R


# --------------------------------------------------------------------------------------------
# the method itself
# --------------------------------------------------------------------------------------------


def test_degree4_spline_without_interior_knots_is_ibatas_quartic():
    """Ibata fits "a fourth order polynomial"; we fit a degree-4 spline. With no interior knots the
    two are the same object, which is what makes the spline a strict generalization."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-50, 80, 60))
    y = 1.0 - 0.02 * x + 3e-4 * x**2 - 1e-6 * x**3 + 2e-8 * x**4 + rng.normal(0, 0.05, 60)
    spline = oo.fit_offset_spline(x, y, x.min(), x.max(), n_interior_knots=0)
    xx = np.linspace(x.min(), x.max(), 200)
    assert np.allclose(spline(xx), np.polyval(np.polyfit(x, y, 4), xx), atol=1e-10)


def test_fit_never_blows_up_between_the_points_it_was_fitted_to():
    """Regression: quantile knots on a clumped stream made the least-squares system near-singular,
    and the fit passed through every binned point while excursing by ~1e290 between them. Medians
    and MADs are blind to that, so it is pinned directly."""
    rng = np.random.default_rng(1)
    # 90 % of the abscissa piled into a narrow clump, as a progenitor remnant does.
    x = np.concatenate([rng.normal(0.0, 0.3, 360), rng.uniform(-30, 40, 40)])
    y = 0.01 * x + rng.normal(0, 0.05, x.size)
    spline = oo.fit_offset_spline(x, y, x.min(), x.max(), n_interior_knots=3)
    xx = np.linspace(x.min(), x.max(), 2000)
    assert np.all(np.isfinite(spline(xx)))
    assert np.max(np.abs(spline(xx))) < 10 * (y.max() - y.min())


def test_project_matches_the_ppc_stack_definition():
    """The simulator's frame math and the diagnostics' frame math must be the same estimator."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_pss", REPO / "scripts" / "ppc_summary_statistics.py"
    )
    pss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pss)

    rng = np.random.default_rng(3)
    R = _random_frame()
    ra = rng.uniform(0, 360, 200)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 200)))
    mura, mudec = rng.normal(0, 5, 200), rng.normal(0, 5, 200)
    for a, b in zip(sf.project(R, ra, dec, mura, mudec), pss.project(R, ra, dec, mura, mudec)):
        assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_deproject_inverts_project():
    rng = np.random.default_rng(4)
    R = _random_frame(1)
    ra = rng.uniform(0, 360, 300)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 300)))
    mura, mudec = rng.normal(0, 5, 300), rng.normal(0, 5, 300)
    phi1, phi2, m1, m2 = sf.project(R, ra, dec, mura, mudec)
    ra_b, dec_b, mura_b, mudec_b = sf.deproject(R, phi1, phi2, m1, m2)
    assert np.abs(((ra_b - ra + 180) % 360) - 180).max() < 1e-9
    assert np.abs(dec_b - dec).max() < 1e-9
    assert np.abs(mura_b - mura).max() < 1e-10
    assert np.abs(mudec_b - mudec).max() < 1e-10


def test_observables_to_cartesian_inverts_sky_projection():
    """The worker emits observables and returns Galactocentric coordinates, which ``simulate`` then
    re-projects. That round trip has to be exact or the stream the network sees is not the stream
    the model generated."""
    rng = np.random.default_rng(5)
    n = 200
    frame = (8.2, 11.1, 245.0, 7.25, 0.0208)
    obs = np.column_stack([
        rng.uniform(0, 360, n),
        np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
        rng.uniform(3, 40, n),
        rng.normal(0, 5, n),
        rng.normal(0, 5, n),
        rng.normal(0, 150, n),
    ])
    xv = oo.observables_to_cartesian(agama, obs, frame)
    back = sky_projection(xv[None], np.asarray(frame)[None])[0]
    assert np.abs(((back[:, 0] - obs[:, 0] + 180) % 360) - 180).max() < 1e-9
    assert np.abs(back[:, 1:] - obs[:, 1:]).max() < 1e-9


def test_distance_modulus_round_trip():
    d = np.geomspace(0.5, 200.0, 50)
    assert np.allclose(oo.distance_from_dm(oo.dm_from_distance(d)), d)


# --------------------------------------------------------------------------------------------
# the simulator
# --------------------------------------------------------------------------------------------


def test_registry_resolves_stream_orbit_offset(sim):
    assert isinstance(sim, OrbitOffsetStreamSimulator)


def test_stripping_history_is_frozen_not_inferred(sim):
    assert sim.local_parameter_names == ["vr", "r", "mu_ra_cosdec", "mu_dec"]
    for key in ("m_progenitor", "a_progenitor", "t_end"):
        assert key not in sim.local_parameter_names
    assert "log10_M200_TwoPowerTriaxial_halo" in sim.global_parameter_names


def test_template_records_the_frozen_locals(sim):
    fiducial = template_fiducial_locals(sim._template_path)
    assert set(fiducial) == set(sim.target_streams.values())
    for stream, j in sim.target_streams.items():
        for key, value in fiducial[j].items():
            spec = sim._priors_local[stream][key]
            assert str(spec["type"]) == "identity"
            assert np.isclose(float(spec["prior_parameters"][0]), value, rtol=1e-6)


def test_shipped_config_passes_the_frozen_prior_guard(sim):
    sim._check_template_matches_priors()


def test_freeing_a_frozen_local_is_rejected():
    """A freed t_end would enter ``local_parameter_names`` and be inferred, though the forward model
    cannot respond to it."""
    register_configs()
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
    s = get_simulator(cfg.simulator)
    if not Path(s._template_path).exists():
        pytest.skip("orbit-offset template not built")
    with pytest.raises(ValueError, match="t_end"):
        s._check_template_matches_priors()


def test_simulate_matches_the_v4_output_contract(sim):
    rng = np.random.default_rng(0)
    out = sim.simulate(sim.sample_prior(4, rng), rng)
    assert out["sim_data_projected"].shape == (4, 2000, 6)
    assert out["sim_data_projected"].dtype == np.float32
    assert out["vcirc_kms"].shape == (4, 19, 1)
    assert np.isfinite(out["vcirc_kms"]).all()
    assert "rho_TwoPowerTriaxial_halo_derived" in out
    assert "a_TwoPowerTriaxial_halo_derived" in out
    # There is no progenitor and no stripping, so reporting a bound mass would be a fabrication.
    assert "m_bound_final" not in out


def test_simulate_produces_usable_rows(sim):
    rng = np.random.default_rng(1)
    out = sim.simulate(sim.sample_prior(6, rng), rng)
    stars = out["sim_data_projected"]
    assert not np.isnan(stars[:, 0, 0]).any(), "rows were dropped by the phi1 coverage guard"
    kept = stars[0][stars[0][:, 0] != -999.0]
    assert kept.shape[0] > 100


def test_sample_compositional_shapes(sim):
    rng = np.random.default_rng(2)
    out = sim.sample_compositional(2, rng)
    m = len(sim.target_streams)
    assert out["j"].shape == (2, m, 1)
    assert out["vr"].shape == (2, m, 1)
    assert out["log10_M200_TwoPowerTriaxial_halo"].shape == (2, 1)   # one potential per group
    assert out["sim_data_projected"].shape == (2, m, 2000, 6)
    assert out["vcirc_kms"].shape == (2, 19, 1)                      # group-level observable


def test_the_stream_responds_to_the_potential(sim):
    """The guard against a surrogate that quietly ignores the parameter it is meant to be
    conditioned on: the stored correction is fixed, so all the potential dependence has to come
    through the re-integrated orbit."""
    rng = np.random.default_rng(3)
    params = sim.sample_prior(1, rng)
    key = "log10_M200_TwoPowerTriaxial_halo"
    tracks = []
    for value in (11.7, 12.3):
        p = {k: np.array(v, copy=True) for k, v in params.items()}
        p[key][:] = value
        out = sim.simulate(p, np.random.default_rng(0))
        a = out["sim_data_projected"][0]
        tracks.append(a[a[:, 0] != -999.0])
    assert min(len(t) for t in tracks) > 100
    assert abs(np.median(tracks[0][:, 1]) - np.median(tracks[1][:, 1])) > 1e-3
