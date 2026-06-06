"""Two Moons simulator + the augmentation reproducibility/stochasticity contract.

These tests pin down the property the augmentation design promises: every augmentation is
*stochastic* (draws change batch to batch, and depend on the seed) yet fully *reproducible* (same
seed -> identical sequence). All randomness must flow through the injected generator, never the
global ``np.random`` state.
"""

from __future__ import annotations

import numpy as np
import pytest

ALL_AUGS = ["gaussian_noise", "multiplicative_noise", "feature_dropout"]
# Strengths that make each augmentation a non-trivial (non-no-op) transform.
STRONG_PARAMS = {
    "noise_key": "x",
    "noise_scale": 0.5,
    "mult_scale": 0.3,
    "dropout_prob": 0.4,
}


def _batch(n=8, n_obs=4, d=2):
    return {"x": np.ones((n, n_obs, d), dtype=np.float32)}


def _build_one(name, seed, params=STRONG_PARAMS):
    """Build a single augmentation through the public registry with a seeded generator."""
    from hydraflow.augmentation.registry import _REGISTRY

    rng = np.random.default_rng(seed)
    # rng.spawn(1) mirrors how build_augmentations isolates each step's stream.
    return _REGISTRY[name](dict(params), rng.spawn(1)[0])


# --------------------------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------------------------- #


def test_all_examples_registered():
    from hydraflow.augmentation.registry import available_augmentations

    for name in ALL_AUGS:
        assert name in available_augmentations()


# --------------------------------------------------------------------------------------------- #
# Stochastic but seed-controlled
# --------------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL_AUGS)
def test_actually_perturbs(name):
    """At non-trivial strength, each augmentation changes the batch."""
    aug = _build_one(name, seed=0)
    out = aug(_batch())
    assert not np.allclose(out["x"], np.ones_like(out["x"]))


@pytest.mark.parametrize("name", ALL_AUGS)
def test_same_seed_is_reproducible(name):
    """Same seed -> bit-identical augmented output."""
    out_a = _build_one(name, seed=123)(_batch())
    out_b = _build_one(name, seed=123)(_batch())
    np.testing.assert_array_equal(out_a["x"], out_b["x"])


@pytest.mark.parametrize("name", ALL_AUGS)
def test_different_seed_differs(name):
    """Different seed -> different draws (the randomness is genuinely seed-controlled)."""
    out_a = _build_one(name, seed=1)(_batch())
    out_b = _build_one(name, seed=2)(_batch())
    assert not np.allclose(out_a["x"], out_b["x"])


@pytest.mark.parametrize("name", ALL_AUGS)
def test_per_batch_stochasticity(name):
    """Consecutive calls on the *same* built augmentation differ (re-drawn every batch)."""
    aug = _build_one(name, seed=7)
    first = aug(_batch())["x"].copy()
    second = aug(_batch())["x"].copy()
    assert not np.allclose(first, second)


@pytest.mark.parametrize("name", ALL_AUGS)
def test_does_not_touch_global_numpy_state(name):
    """Randomness comes only from the injected generator, not global np.random."""
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    _build_one(name, seed=42)(_batch())
    after = np.random.get_state()[1]
    np.testing.assert_array_equal(before, after)


# --------------------------------------------------------------------------------------------- #
# build_augmentations: per-step independent, order-insensitive streams
# --------------------------------------------------------------------------------------------- #


def _compose_aug(compose, steps, seed):
    from hydraflow.augmentation.registry import build_augmentations

    cfg = compose(
        [
            f"augmentation.steps=[{','.join(steps)}]",
            "+augmentation.params.noise_scale=0.5",
            "+augmentation.params.mult_scale=0.3",
            "+augmentation.params.dropout_prob=0.4",
        ]
    )
    return build_augmentations(cfg.augmentation, np.random.default_rng(seed))


def test_build_augmentations_full_chain_reproducible(compose):
    """Same seed + same step list -> identical end-to-end result through build_augmentations."""

    def run(seed):
        out = _batch()
        for a in _compose_aug(compose, ALL_AUGS, seed):
            out = a(out)
        return out["x"]

    np.testing.assert_array_equal(run(5), run(5))


def test_per_step_streams_are_independent(compose):
    """A step's random stream is its own spawn child, so it doesn't depend on trailing steps."""
    first_alone = _compose_aug(compose, ["gaussian_noise"], 5)[0](_batch())["x"]
    first_with_more = _compose_aug(compose, ALL_AUGS, 5)[0](_batch())["x"]
    np.testing.assert_array_equal(first_alone, first_with_more)


# --------------------------------------------------------------------------------------------- #
# Two Moons simulator
# --------------------------------------------------------------------------------------------- #


def test_two_moons_registered():
    from hydraflow.simulators.registry import available_simulators

    assert "two_moons" in available_simulators()


def test_two_moons_shapes_and_reproducibility():
    from hydraflow.simulators.registry import get_simulator

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
    assert out_a["theta2"].shape == (16, 1)
    assert out_a["x"].shape == (16, 5, 2)  # (n, n_obs, 2)

    # Same seed -> identical; different seed -> different (stochastic, seed-controlled).
    np.testing.assert_array_equal(out_a["x"], out_b["x"])
    assert not np.allclose(out_a["x"], out_c["x"])
