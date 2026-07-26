"""Tests for the differentiable Lotka-Volterra core (no BayesFlow / Keras needed)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(scope="module")
def lv():
    from hydrabflow.simulators import lv_jax

    return lv_jax


def test_selftest_passes(lv):
    lv._selftest()


def test_config_validation(lv):
    # t_end/dt must be divisible by n_obs so observation times land on solver steps.
    with pytest.raises(ValueError, match="divisible"):
        lv.make_config({"t_end": 20.0, "dt": 0.05, "n_obs": 3})
    with pytest.raises(ValueError, match="positive"):
        lv.make_config({"prior_std": [0.5, 0.5, 0.0, 0.5]})
    with pytest.raises(ValueError, match="4 entries"):
        lv.make_config({"prior_mean": [0.0, 0.0]})


def test_rk4_matches_scipy(lv):
    """The hand-rolled RK4 must agree with a high-accuracy reference solver.

    The tolerance (1e-3 relative) is far tighter than needed: the observation noise the likelihood is
    built around is 1e-1, so float32 RK4 error is irrelevant by ~3 orders of magnitude. This is the
    check that justifies not enabling jax_enable_x64 globally.
    """
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp

    cfg = lv.make_config()
    rng = np.random.default_rng(0)
    for theta in lv.sample_prior(3, rng, cfg):
        alpha, beta, gamma, delta = np.exp(theta)
        sol = solve_ivp(
            lambda _t, s: [alpha * s[0] - beta * s[0] * s[1], -gamma * s[1] + delta * s[0] * s[1]],
            (0.0, cfg.t_end),
            [cfg.x0, cfg.y0],
            t_eval=cfg.obs_times,
            rtol=1e-10,
            atol=1e-12,
        )
        mine = np.asarray(lv.lv_states(jnp.asarray(theta, dtype=jnp.float32), cfg))
        assert np.max(np.abs(mine - sol.y.T) / np.abs(sol.y.T)) < 1e-3


def test_gradient_matches_finite_differences(lv):
    cfg = lv.make_config()
    theta_true = jnp.asarray(cfg.prior_mean)
    x_obs = lv.lv_states(theta_true, cfg)
    probe = theta_true + 0.05  # off-optimum, where the gradient is O(100) and FD is reliable

    grad = lv.grad_log_likelihood(probe, x_obs, cfg)
    eps = 1e-2
    for i in range(4):
        fd = (
            lv.log_likelihood(probe.at[i].add(eps), x_obs, cfg)
            - lv.log_likelihood(probe.at[i].add(-eps), x_obs, cfg)
        ) / (2 * eps)
        assert abs(float(fd) - float(grad[i])) / max(1.0, abs(float(fd))) < 5e-2


def test_gradient_vanishes_at_noise_free_truth(lv):
    """With x_obs == states(theta), the residual is identically 0, so the gradient is exactly 0."""
    cfg = lv.make_config()
    theta = jnp.asarray(cfg.prior_mean)
    grad = lv.grad_log_likelihood(theta, lv.lv_states(theta, cfg), cfg)
    assert float(jnp.max(jnp.abs(grad))) == 0.0


def test_wild_parameters_give_finite_gradients(lv):
    """The state clamp must keep out-of-prior evaluations finite (the 'bad gradients' pitfall).

    Guidance evaluates the model at the diffusion model's denoised estimate, which early in the
    reverse process is far outside the prior. Unclamped, the LV ODE overflows to inf within a few
    steps and every downstream gradient becomes NaN.
    """
    cfg = lv.make_config()
    x_obs = lv.lv_states(jnp.asarray(cfg.prior_mean), cfg)
    for value in (8.0, -8.0, 15.0):
        theta = jnp.full((4,), value)
        assert bool(jnp.all(jnp.isfinite(lv.lv_states(theta, cfg))))
        assert bool(jnp.all(jnp.isfinite(lv.grad_log_likelihood(theta, x_obs, cfg))))


def test_soft_clip_is_bounded_and_smooth(lv):
    cfg = lv.make_config()
    mean = np.asarray(cfg.prior_mean)
    std = np.asarray(cfg.prior_std)
    n_std = 4.0

    def grad_at(point):
        # Each parameter has its own prior mean, so the probe must be a vector: filling all four
        # coordinates with one scalar would sit far from the mean of the others.
        point = np.broadcast_to(np.asarray(point, dtype=np.float32), (4,))
        return np.asarray(
            jax.grad(lambda t: jnp.sum(lv.soft_clip_theta(t, cfg, n_std)))(jnp.asarray(point))
        )

    # Bounded for absurd inputs, in both directions.
    for value in (50.0, -50.0):
        clipped = np.asarray(lv.soft_clip_theta(jnp.full((4,), value), cfg, n_std))
        assert np.all(clipped <= mean + n_std * std + 1e-5)
        assert np.all(clipped >= mean - n_std * std - 1e-5)

    # Near the prior mean: close to the identity, with an O(1) derivative. This is the regime
    # guidance actually operates in.
    assert np.allclose(np.asarray(lv.soft_clip_theta(jnp.asarray(mean), cfg, n_std)), mean, atol=1e-5)
    assert np.allclose(grad_at(mean), 1.0, atol=1e-3)

    # Just outside the box the derivative is reduced but still non-zero (smooth, unlike a hard clip).
    edge = grad_at(mean + n_std * std)
    assert np.all(edge > 0.0)
    assert np.all(edge < 1.0)

    # Far outside, tanh saturates and the derivative underflows to exactly 0 in float32 — intended:
    # absurd denoised estimates contribute no guidance. Documented in soft_clip_theta.
    assert np.all(grad_at(50.0) == 0.0)


def test_batched_paths_agree_with_single(lv):
    cfg = lv.make_config()
    rng = np.random.default_rng(1)
    batch = jnp.asarray(lv.sample_prior(5, rng, cfg), dtype=jnp.float32)
    x_obs = lv.lv_states(jnp.asarray(cfg.prior_mean), cfg)

    states = lv.lv_states_batch(batch, cfg)
    grads = lv.grad_log_likelihood_batch(batch, x_obs, cfg)
    lls = lv.log_likelihood_batch(batch, x_obs, cfg)
    assert states.shape == (5, cfg.n_obs, 2)
    for i in range(5):
        assert jnp.allclose(states[i], lv.lv_states(batch[i], cfg), rtol=1e-5)
        assert jnp.allclose(grads[i], lv.grad_log_likelihood(batch[i], x_obs, cfg), rtol=1e-4)
        assert jnp.allclose(lls[i], lv.log_likelihood(batch[i], x_obs, cfg), rtol=1e-5)


def test_prior_sampling_is_reproducible(lv):
    cfg = lv.make_config()
    a = lv.sample_prior(16, np.random.default_rng(7), cfg)
    b = lv.sample_prior(16, np.random.default_rng(7), cfg)
    assert np.array_equal(a, b)
    assert a.shape == (16, 4)
    # Moments should match the configured prior.
    big = lv.sample_prior(20000, np.random.default_rng(0), cfg)
    assert np.allclose(big.mean(axis=0), cfg.prior_mean, atol=0.02)
    assert np.allclose(big.std(axis=0), cfg.prior_std, atol=0.02)
