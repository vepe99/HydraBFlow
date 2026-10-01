"""Exact diffused-prior score (pipeline.diffused_prior) vs Monte Carlo, for the rc38 prior family."""

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
enable_x64 = lambda: jax.enable_x64(True)  # noqa: E731  (context manager, jax >= 0.10)

from hydrabflow.pipeline.diffused_prior import (  # noqa: E402
    _diffused_callable, make_log_prob_t, make_prior_score_t,
)
from hydrabflow.simulators.stream_common import MOSTER13, mu_moster, sample_stream_prior  # noqa: E402

L, LS = "log10_M200_TwoPowerTriaxial_halo", "log10_Mstar"
SPEC = {
    "gamma_TwoPowerTriaxial_halo": {"type": "uniform", "prior_parameters": [0.0, 1.5]},
    "q_TwoPowerTriaxial_halo": {"type": "uniform", "prior_parameters": [0.5, 1.5]},
    L: {"type": "uniform", "prior_parameters": [11.5, 12.5]},
    "ln_cvprime_TwoPowerTriaxial_halo": {"type": "normal", "prior_parameters": [2.56, 0.272]},
    LS: {"type": "moster_conditional", "prior_parameters": [0.2], "condition_on": L},
    "ln_R_d_thin": {"type": "normal", "prior_parameters": [math.log(2.6), 0.2]},
    "ln_R_d_thick": {"type": "normal", "prior_parameters": [math.log(3.6), 0.2]},
    "ln_f_thick": {"type": "normal", "prior_parameters": [math.log(0.12), 0.1]},
}
NAMES = list(SPEC)
# 1e-4 = the EDM sampler's sigma_min; 0.99992 = its sigma_max=80 in the variance-preserving
# parameterization (alpha = sqrt(sigmoid(-2 ln 80)) = 0.0125; sigma = 1 would mean alpha = 0).
SIGMAS = [1e-4, 1e-3, 3e-2, 0.3, 0.99, 0.99992]


def _alpha(sigma):  # variance-preserving schedule
    return math.sqrt(1.0 - sigma**2)


def _log_p0(x0):
    """Clean prior log density of all 8 parameters, (M, 8) -> (M,), in theta space."""
    out = np.zeros(len(x0))
    for i, n in enumerate(NAMES):
        s = SPEC[n]
        if s["type"] == "uniform":
            a, b = s["prior_parameters"]
            out += np.where((x0[:, i] >= a) & (x0[:, i] <= b), -np.log(b - a), -np.inf)
        elif s["type"] == "normal":
            m, sd = s["prior_parameters"]
            out += -0.5 * ((x0[:, i] - m) / sd) ** 2 - np.log(sd)
        else:
            mu = mu_moster(x0[:, NAMES.index(L)], **MOSTER13)
            out += -0.5 * ((x0[:, i] - mu) / 0.2) ** 2
    return out


def _is_score(x, alpha, sigma, dims, n_pairs=400_000, seed=0, log_p0=_log_p0,
              ref_point=None):
    """Importance-sampled grad log p_t(x): proposal x0 = x/alpha + (sigma/alpha) eps (which is the
    likelihood N(x; alpha x0, sigma^2) in x0), weights = p0(x0); antithetic pairs +-eps cancel the
    O(1/sigma) noise so the estimate stays accurate at sigma = 1e-4. Only ``dims`` are perturbed
    (the others are fixed at x/alpha, valid because the prior factorizes over the other blocks)."""
    rng = np.random.default_rng(seed)
    eps = np.zeros((n_pairs, len(NAMES)))
    eps[:, dims] = rng.normal(size=(n_pairs, len(dims)))
    # The prior factorizes over blocks, so the dims outside ``dims`` only add a constant to
    # log p0: pin them at an in-support point (x/alpha can leave the support at large sigma).
    base = np.array(ref_point if ref_point is not None else x / alpha, dtype=float)
    base[dims] = (x / alpha)[dims]
    lp = log_p0(base + sigma / alpha * eps)
    lm = log_p0(base - sigma / alpha * eps)
    mx = max(lp.max(), lm.max())
    wp, wm = np.exp(lp - mx), np.exp(lm - mx)
    num = ((wp - wm)[:, None] * eps).sum(0)
    return num / (wp + wm).sum() / sigma  # (alpha E[x0|x] - x)/sigma^2


def _point(sigma, seed=1):
    """A diffused point near a typical prior draw (interior of every uniform)."""
    x0 = np.array([0.7, 1.1, 12.1, 2.5, 0.0, math.log(2.4), math.log(3.9), math.log(0.1)])
    x0[4] = mu_moster(12.1, **MOSTER13) + 0.1
    return _alpha(sigma) * x0 + sigma * np.random.default_rng(seed).normal(size=8) * 0.5


@pytest.mark.parametrize("sigma", SIGMAS)
def test_score_matches_importance_sampling(sigma):
    alpha = _alpha(sigma)
    x = _point(sigma)
    with enable_x64():
        score = np.asarray(make_prior_score_t(SPEC, NAMES)(jnp.asarray(x[None]), alpha, sigma))[0]
    for dims in ([0], [1], [2, 4]):  # uniform, uniform, the (M200, Mstar) block
        ref = _is_score(x, alpha, sigma, dims, ref_point=_point(0.0))
        for d in dims:
            assert score[d] == pytest.approx(ref[d], rel=0.03, abs=0.02 * (1 + abs(ref[d]))), (d, sigma)
    # Gaussian block: closed form
    for i in (3, 5, 6, 7):
        m, s = SPEC[NAMES[i]]["prior_parameters"]
        assert score[i] == pytest.approx(-(x[i] - alpha * m) / (alpha**2 * s**2 + sigma**2), rel=1e-9)


def test_small_sigma_limit_is_the_clean_score():
    """At sigma = 1e-4 inside the support, the diffused score is the clean prior score."""
    sigma, alpha = 1e-4, _alpha(1e-4)
    lm, ls = 12.1, mu_moster(12.1, **MOSTER13) + 0.1
    x = np.array([0.7, 1.1, lm, 2.5, ls, 1.0, 1.3, -2.0])
    with enable_x64():
        s = np.asarray(make_prior_score_t(SPEC, NAMES)(jnp.asarray(alpha * x[None]), alpha, sigma))[0]
        dmu = float(jax.grad(lambda v: mu_moster(v, xp=jnp, **MOSTER13))(lm))
    r = (ls - mu_moster(lm, **MOSTER13)) / 0.2**2
    assert s[4] == pytest.approx(-r, rel=1e-3)
    assert s[2] == pytest.approx(r * dmu, rel=1e-3)
    assert abs(s[0]) < 1e-6 and abs(s[1]) < 1e-6


@pytest.mark.parametrize("sigma", [0.3, 0.99])
def test_score_matches_tweedie_on_prior_draws(sigma):
    """Independent check: self-normalized Tweedie estimate from plain prior draws."""
    alpha = _alpha(sigma)
    draws = sample_stream_prior(SPEC, {"S": {}}, {"S": 0}, 400_000, np.random.default_rng(3))
    x0 = np.concatenate([draws[n] for n in NAMES], axis=1)
    x = _point(sigma, seed=4)
    logw = -0.5 * (((x[None] - alpha * x0) / sigma) ** 2).sum(1)
    w = np.exp(logw - logw.max())
    ref = (alpha * (w[:, None] * x0).sum(0) / w.sum() - x) / sigma**2
    with enable_x64():
        score = np.asarray(make_prior_score_t(SPEC, NAMES)(jnp.asarray(x[None]), alpha, sigma))[0]
    np.testing.assert_allclose(score, ref, rtol=0.05, atol=0.05)


def test_block_c_marginal_matches_prior_histogram():
    """At small t the diffused (M200, Mstar) density integrates to the prior's Mstar marginal."""
    sigma = 1e-3
    alpha = _alpha(sigma)
    spec = {L: SPEC[L], LS: SPEC[LS]}
    x1 = np.linspace(11.45, 12.55, 2201)
    x2 = np.linspace(9.2, 11.6, 121)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    with enable_x64():
        lp = np.asarray(make_log_prob_t(spec, [L, LS])(
            jnp.asarray(np.column_stack([X1.ravel(), X2.ravel()])), alpha, sigma))
    p = np.exp(lp).reshape(X1.shape)
    marg2 = np.trapezoid(p, x1, axis=0)  # density of the diffused Mstar coordinate
    assert np.trapezoid(marg2, x2) == pytest.approx(1.0, abs=2e-3)
    marg1 = np.trapezoid(p, x2, axis=1)
    inside = (x1 > 11.52) & (x1 < 12.48)
    np.testing.assert_allclose(marg1[inside], 1.0, atol=5e-3)  # uniform parent survives
    rng = np.random.default_rng(5)
    lm = rng.uniform(11.5, 12.5, 2_000_000)
    ls = mu_moster(lm, **MOSTER13) + 0.2 * rng.normal(size=lm.size)
    hist, edges = np.histogram(alpha * ls + sigma * rng.normal(size=lm.size), bins=24,
                               range=(9.4, 11.4), density=False)
    hist = hist / (lm.size * np.diff(edges))
    centres = 0.5 * (edges[1:] + edges[:-1])
    np.testing.assert_allclose(np.interp(centres, x2, marg2), hist, rtol=0.03, atol=5e-3)


@pytest.mark.parametrize("sigma", [1e-3, 0.3])
def test_standardized_space_matches_importance_sampling_and_jits(sigma):
    """In z = (theta - mean)/std the diffused prior is that of the transformed prior
    p_z(z) = p_theta(mean + std z) prod(std); check the jitted z-space score against IS in z."""
    mean = np.array([0.75, 1.0, 12.0, 2.56, 10.3, 0.95, 1.28, -2.12])
    std = np.array([0.43, 0.29, 0.29, 0.27, 0.3, 0.2, 0.2, 0.1])
    alpha = _alpha(sigma)
    z = alpha * (_point(0.0, seed=7) - mean) / std + sigma * 0.3
    with enable_x64():
        fn = jax.jit(make_prior_score_t(SPEC, NAMES, mean=mean, std=std))
        score = np.asarray(fn(jnp.asarray(z[None]), alpha, sigma))[0]
    log_pz = lambda zz: _log_p0(mean + std * zz)  # noqa: E731
    for dims in ([0], [1], [2, 4], [3], [5]):
        ref = _is_score(z, alpha, sigma, dims, log_p0=log_pz)
        for d in dims:
            assert score[d] == pytest.approx(ref[d], rel=0.03, abs=0.02 * (1 + abs(ref[d]))), (d, sigma)


class _FakeSchedule:
    def get_log_snr(self, t, training):
        return 2.0 - 8.0 * jnp.asarray(t)

    def get_alpha_sigma(self, log_snr):
        return jnp.sqrt(jax.nn.sigmoid(log_snr)), jnp.sqrt(jax.nn.sigmoid(-log_snr))


def test_bayesflow_callable_returns_theta_space_score_over_std():
    mean = np.linspace(-1, 1, 8)
    std = np.linspace(0.5, 2.0, 8)
    fn = _diffused_callable(SPEC, NAMES, mean, std, _FakeSchedule())
    theta = _point(0.0, seed=8)
    out = fn({n: jnp.asarray([[theta[i]]]) for i, n in enumerate(NAMES)}, time=0.3)
    ls = 2.0 - 8.0 * 0.3
    a, s = math.sqrt(1 / (1 + math.exp(-ls))), math.sqrt(1 / (1 + math.exp(ls)))
    z = (theta - mean) / std
    sz = np.asarray(make_prior_score_t(SPEC, NAMES, mean=mean, std=std)(jnp.asarray(z[None]), a, s))[0]
    got = np.array([float(out[n][0, 0]) for n in NAMES])
    np.testing.assert_allclose(got, sz / std, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("sigma", [1e-4, 0.99992])
def test_float32_matches_float64(sigma):
    """BayesFlow samples in float32: the score must survive it at both ends of the schedule."""
    alpha = _alpha(sigma)
    mean = np.array([0.75, 1.0, 12.0, 2.56, 10.3, 0.95, 1.28, -2.12])
    std = np.array([0.43, 0.29, 0.29, 0.27, 0.3, 0.2, 0.2, 0.1])
    z = alpha * (_point(0.0, seed=9) - mean) / std + sigma * 0.3
    fn = make_prior_score_t(SPEC, NAMES, mean=mean, std=std)
    s32 = np.asarray(fn(jnp.asarray(z[None], dtype=jnp.float32), alpha, sigma))[0]
    with enable_x64():
        s64 = np.asarray(fn(jnp.asarray(z[None]), alpha, sigma))[0]
    assert np.all(np.isfinite(s32))
    np.testing.assert_allclose(s32, s64, rtol=2e-3, atol=1e-3 * (1 + np.abs(s64).max()))
