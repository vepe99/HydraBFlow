"""Per-draw simulators (``is_batched = False``) and the shape/name guard in ``BaseSimulator``."""

from __future__ import annotations

import numpy as np
import pytest

from hydrabflow.simulators.base import BaseSimulator


class _Batched(BaseSimulator):
    """A toy forward model. Deterministic, so the batched and per-draw paths must agree exactly."""

    parameter_names = ["a"]
    observable_keys = ["y"]

    def sample_prior(self, n, rng):
        return {"a": rng.uniform(size=(n, 1))}

    def simulate(self, theta, rng):
        return {"y": np.asarray(theta["a"]) * np.arange(3.0)}  # (n, 3)


class _PerDraw(_Batched):
    is_batched = False

    def simulate(self, theta, rng):
        assert np.shape(theta["a"]) == (1,), "per-draw simulate gets one row, shape (1,)"
        return {"y": theta["a"].item() * np.arange(3.0)}  # (3,)


def test_per_draw_matches_batched():
    out_b = _Batched().sample(8, np.random.default_rng(0))
    out_p = _PerDraw().sample(8, np.random.default_rng(0))

    assert out_b["a"].shape == out_p["a"].shape == (8, 1)
    assert out_b["y"].shape == out_p["y"].shape == (8, 3)
    np.testing.assert_array_equal(out_b["y"], out_p["y"])


def test_declared_name_mismatch_raises():
    class _Misnamed(_Batched):
        parameter_names = ["alpha"]  # sample_prior actually returns "a"

    with pytest.raises(ValueError, match="must return"):
        _Misnamed().sample(4, np.random.default_rng(0))


def test_wrong_leading_axis_raises():
    class _TimeMajor(_Batched):
        def simulate(self, theta, rng):
            return {"y": np.zeros((3, len(theta["a"])))}  # transposed: (event, n)

    with pytest.raises(ValueError, match="leading axis"):
        _TimeMajor().sample(4, np.random.default_rng(0))


def test_sir_shapes(compose):
    """The real per-draw simulator, through the real config."""
    from hydrabflow.registry import get_simulator

    cfg = compose(["simulator=sir"], fill=False)
    sim = get_simulator(cfg.simulator)

    out = sim.sample(4, np.random.default_rng(0))
    assert out["cases"].shape == (4, cfg.simulator.params.T, 1)
    assert all(out[k].shape == (4, 1) for k in sim.parameter_names)
