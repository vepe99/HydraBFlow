"""Progenitor mass loss in the particle-spray forward model."""

from __future__ import annotations

import numpy as np
import pytest

from hydrabflow.simulators.stream_agama import _mass_track, _spray_stream

agama = pytest.importorskip("agama")


def test_mass_track_endpoints_and_monotonicity():
    t = np.linspace(-4.0, 0.0, 9)
    for law in ("linear", "exponential"):
        m = _mass_track(1.0e5, 2.0e4, t, 4.0, law)
        assert m[0] == pytest.approx(1.0e5)
        assert m[-1] == pytest.approx(2.0e4)
        assert np.all(np.diff(m) < 0.0)
    lin = _mass_track(1.0e5, 2.0e4, t, 4.0, "linear")
    exp = _mass_track(1.0e5, 2.0e4, t, 4.0, "exponential")
    # exponential loses its mass earlier, so it sits below the linear track in between
    assert np.all(exp[1:-1] < lin[1:-1])


def test_mass_track_no_loss_when_final_exceeds_initial():
    t = np.linspace(-4.0, 0.0, 5)
    assert np.allclose(_mass_track(3.0e4, 5.0e4, t, 4.0, "linear"), 3.0e4)


def test_mass_track_rejects_unknown_law():
    with pytest.raises(ValueError, match="unknown mass-loss law"):
        _mass_track(1.0e5, 1.0e4, np.linspace(-1, 0, 3), 1.0, "quadratic")


def _pot(ag):
    ag.setUnits(mass=1, length=1, velocity=1)
    return ag.Potential(type="NFW", mass=1e12, scaleRadius=15.0)


def _stream(ag, **kw):
    return _spray_stream(
        ag, _pot(ag), np.array([10.0, 0.0, 0.0, 0.0, 200.0, 0.0]),
        mass_sat=1.0e5, radius_sat=0.02, time_total=4.0 / 0.9778,
        num_particles=200, rng=np.random.default_rng(0), method="chen", **kw,
    )


def test_no_mass_loss_reproduces_fixed_mass_path_bitwise():
    """``mass_final=None`` must leave the original fixed-mass spray untouched."""
    a = _stream(agama)
    b = _stream(agama, mass_final=None)
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_mass_loss_produces_a_colder_narrower_stream():
    """Shedding mass shrinks the Jacobi radius and the escape speed, so the tails are tighter."""
    fixed = _stream(agama)
    lost = _stream(agama, mass_final=1.0e4, mass_loss="linear")
    ok = np.isfinite(fixed).all(1) & np.isfinite(lost).all(1)
    assert ok.sum() > 50
    r_fixed = np.linalg.norm(fixed[ok, :3], axis=1)
    r_lost = np.linalg.norm(lost[ok, :3], axis=1)
    assert not np.allclose(r_fixed, r_lost)
    # velocity spread about the progenitor's orbit is set by v_j ~ M^(1/3) and v_esc ~ M^(1/2)
    assert np.std(lost[ok, 3:]) < np.std(fixed[ok, 3:])


def test_mass_loss_law_changes_the_realization():
    lin = _stream(agama, mass_final=1.0e4, mass_loss="linear")
    exp = _stream(agama, mass_final=1.0e4, mass_loss="exponential")
    ok = np.isfinite(lin).all(1) & np.isfinite(exp).all(1)
    assert ok.sum() > 50
    assert not np.allclose(lin[ok], exp[ok])
