"""Two Moons simulator + the augmentation reproducibility/stochasticity contract.

These tests pin down the property the augmentation design promises: an augmentation is *stochastic*
(draws change batch to batch, and depend on the seed) yet fully *reproducible* (same seed ->
identical sequence). All randomness must flow through the injected generator, never global numpy.
"""

from __future__ import annotations

import numpy as np

STRONG_PARAMS = {"noise_key": "x", "noise_scale": 0.5}  # non-no-op strength


def _batch(n=8, n_obs=4, d=2):
    return {"x": np.ones((n, n_obs, d), dtype=np.float32)}


def _build_one(seed, params=STRONG_PARAMS):
    """Build the shipped augmentation through the registry with a seeded generator."""
    from hydrabflow.registry import AUGMENTATIONS

    rng = np.random.default_rng(seed)
    # rng.spawn(1) mirrors how build_augmentations isolates each step's stream.
    return AUGMENTATIONS.get("gaussian_noise")(dict(params), rng.spawn(1)[0], {})


def test_actually_perturbs():
    out = _build_one(seed=0)(_batch())
    assert not np.allclose(out["x"], np.ones_like(out["x"]))


def test_no_op_at_default_strength():
    """The shipped config trains without augmentation: zero scale must leave the batch alone."""
    out = _build_one(seed=0, params={})(_batch())
    np.testing.assert_array_equal(out["x"], np.ones_like(out["x"]))


def test_same_seed_is_reproducible():
    np.testing.assert_array_equal(
        _build_one(seed=123)(_batch())["x"], _build_one(seed=123)(_batch())["x"]
    )


def test_different_seed_differs():
    assert not np.allclose(_build_one(seed=1)(_batch())["x"], _build_one(seed=2)(_batch())["x"])


def test_per_batch_stochasticity():
    """Consecutive calls on the *same* built augmentation differ (re-drawn every batch)."""
    aug = _build_one(seed=7)
    first = aug(_batch())["x"].copy()
    second = aug(_batch())["x"].copy()
    assert not np.allclose(first, second)


def test_does_not_touch_global_numpy_state():
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    _build_one(seed=42)(_batch())
    np.testing.assert_array_equal(before, np.random.get_state()[1])


def test_build_augmentations_end_to_end_reproducible(compose):
    """Same seed + same step list -> identical result through the public builder."""
    from hydrabflow.registry import build_augmentations

    def run(seed):
        cfg = compose(
            ["augmentation.steps=[gaussian_noise]", "+augmentation.params.noise_scale=0.5"]
        )
        out = _batch()
        for aug in build_augmentations(cfg.augmentation, np.random.default_rng(seed)):
            out = aug(out)
        return out["x"]

    np.testing.assert_array_equal(run(5), run(5))
    assert not np.allclose(run(5), run(6))


# --------------------------------------------------------------------------------------------- #
# Two Moons simulator
# --------------------------------------------------------------------------------------------- #


def test_two_moons_shapes_and_reproducibility():
    from hydrabflow.registry import get_simulator

    class _Cfg:
        name = "two_moons"
        params = {"n_obs": 5}

    sim = get_simulator(_Cfg())
    assert sim.parameter_names == ["theta1", "theta2"]
    assert sim.observable_keys == ["x"]

    out_a = sim.sample(16, np.random.default_rng(0))
    out_b = sim.sample(16, np.random.default_rng(0))
    out_c = sim.sample(16, np.random.default_rng(1))

    assert out_a["theta1"].shape == (16, 1)
    assert out_a["x"].shape == (16, 5, 2)  # (n, n_obs, 2)

    # Same seed -> identical; different seed -> different (stochastic, seed-controlled).
    np.testing.assert_array_equal(out_a["x"], out_b["x"])
    assert not np.allclose(out_a["x"], out_c["x"])
