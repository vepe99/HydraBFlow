"""Differentiable Lotka-Volterra forward model in pure JAX.

This module is the **single source of truth** for the LV testbed used by the simulator-gradient
guidance study. Three consumers share it, and they must agree exactly or the study is meaningless:

* ``simulators.lotka_volterra`` — dataset generation (the ``BaseSimulator`` wrapper),
* ``networks.guided_diffusion`` — the guidance term ``grad_log_likelihood``,
* ``pipeline.reference`` — the MALA reference posterior via ``log_posterior``.

Model
-----
Predator-prey ODE with four positive rates::

    dx/dt =  alpha * x - beta  * x * y      (prey)
    dy/dt = -gamma * y + delta * x * y      (predator)

**Parameters are inferred in log space** (``log_alpha, log_beta, log_gamma, log_delta``) with
independent Normal priors. This is deliberate and load-bearing for the guidance study:

* the support is unbounded, which is what a diffusion model over ``R^4`` wants — no positivity
  constraint to fight and no bounded-domain preprocessing step,
* the prior is roughly unit-scale, so the network needs no parameter standardization (see
  ``conf/training/lv.yaml``, which standardizes only ``summary_variables``) and the guidance
  gradient therefore lives in the *same* space as the diffusion state,
* ``exp`` is smooth, so ``d log p / d log theta`` is well conditioned.

The prior is the log of the ``sbibm`` Lotka-Volterra prior: ``LogNormal(-0.125, 0.5)`` for the two
"linear" rates and ``LogNormal(-3, 0.5)`` for the two "interaction" rates.

Observation model
-----------------
The ODE is solved with fixed-step RK4 and subsampled at ``n_obs`` evenly spaced times, then
corrupted by **multiplicative lognormal** noise with scale ``obs_sigma``::

    x_obs = states * exp(obs_sigma * eps),   eps ~ Normal(0, 1)

so the log-likelihood is a Gaussian in ``log x_obs`` around ``log states``.

Precision
---------
Everything runs in the ambient JAX dtype (float32 by default). The global ``jax_enable_x64`` flag
is deliberately **not** set: it is process-wide and would change dtypes underneath Keras/BayesFlow.
It buys nothing here — ``obs_sigma`` is 1e-1 while the float32 RK4 truncation/rounding error is
~1e-5 relative (verified against ``scipy.integrate.solve_ivp`` in ``tests/test_lv_jax.py``), i.e.
four orders of magnitude below the noise the likelihood is built around.
"""

from __future__ import annotations

import functools
from typing import Mapping, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

#: Inference parameter names, in the order they are concatenated into ``inference_variables``.
#: The guidance term relies on this order matching the last axis of the diffusion state.
PARAMETER_NAMES = ["log_alpha", "log_beta", "log_gamma", "log_delta"]

#: log of the sbibm LogNormal prior: (alpha, beta, gamma, delta).
DEFAULT_PRIOR_MEAN = (-0.125, -3.0, -0.125, -3.0)
DEFAULT_PRIOR_STD = (0.5, 0.5, 0.5, 0.5)


class LVConfig(NamedTuple):
    """Static (hashable) forward-model settings, so it can be a ``jax.jit`` static argument."""

    x0: float = 30.0
    y0: float = 1.0
    t_end: float = 20.0
    dt: float = 0.05
    n_obs: int = 10
    obs_sigma: float = 0.1
    prior_mean: Tuple[float, ...] = DEFAULT_PRIOR_MEAN
    prior_std: Tuple[float, ...] = DEFAULT_PRIOR_STD
    #: States are clamped into ``[state_min, state_max]`` at every RK4 stage. Guidance evaluates
    #: the model at denoised estimates that can be far outside the prior early in the reverse
    #: diffusion; without a clamp those blow up to inf and poison the gradient (the clamp yields a
    #: zero gradient there instead, which the caller then detects and gates away).
    state_min: float = 1e-6
    state_max: float = 1e6

    @property
    def n_steps(self) -> int:
        return int(round(self.t_end / self.dt))

    @property
    def obs_stride(self) -> int:
        return self.n_steps // self.n_obs

    @property
    def obs_times(self) -> np.ndarray:
        return np.arange(1, self.n_obs + 1) * self.obs_stride * self.dt


def make_config(params: Mapping | None = None) -> LVConfig:
    """Build an :class:`LVConfig` from a free-form ``simulator.params`` mapping.

    Values arrive as an OmegaConf node, so every field is cast explicitly.
    """
    params = dict(params or {})

    def _f(key: str, default: float) -> float:
        return float(params.get(key, default))

    def _tuple(key: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
        value = params.get(key, None)
        if value is None:
            return default
        out = tuple(float(v) for v in value)
        if len(out) != 4:
            raise ValueError(f"simulator.params.{key} must have 4 entries, got {len(out)}")
        return out

    cfg = LVConfig(
        x0=_f("x0", 30.0),
        y0=_f("y0", 1.0),
        t_end=_f("t_end", 20.0),
        dt=_f("dt", 0.05),
        n_obs=int(params.get("n_obs", 10)),
        obs_sigma=_f("obs_sigma", 0.1),
        prior_mean=_tuple("prior_mean", DEFAULT_PRIOR_MEAN),
        prior_std=_tuple("prior_std", DEFAULT_PRIOR_STD),
        state_min=_f("state_min", 1e-6),
        state_max=_f("state_max", 1e6),
    )
    if cfg.n_steps % cfg.n_obs != 0:
        raise ValueError(
            f"t_end/dt = {cfg.n_steps} solver steps must be divisible by n_obs = {cfg.n_obs} so "
            "observation times land exactly on solver steps; adjust dt, t_end or n_obs."
        )
    if any(s <= 0 for s in cfg.prior_std):
        raise ValueError(f"simulator.params.prior_std must be positive, got {cfg.prior_std}")
    return cfg


# --------------------------------------------------------------------------------------------- #
# Forward model
# --------------------------------------------------------------------------------------------- #


def _rhs(state: jnp.ndarray, rates: jnp.ndarray) -> jnp.ndarray:
    """Lotka-Volterra right-hand side. ``state`` is ``(2,)``, ``rates`` is ``(alpha,beta,gamma,delta)``."""
    x, y = state[0], state[1]
    alpha, beta, gamma, delta = rates[0], rates[1], rates[2], rates[3]
    return jnp.stack([alpha * x - beta * x * y, -gamma * y + delta * x * y])


@functools.partial(jax.jit, static_argnames="cfg")
def lv_states(theta_log: jnp.ndarray, cfg: LVConfig) -> jnp.ndarray:
    """Noise-free trajectory at the ``n_obs`` observation times for one parameter vector.

    Parameters
    ----------
    theta_log : array of shape ``(4,)``
        ``[log_alpha, log_beta, log_gamma, log_delta]``.

    Returns
    -------
    array of shape ``(n_obs, 2)``
        Prey/predator abundance at each observation time, clamped to be strictly positive.
    """
    rates = jnp.exp(theta_log)
    dt = jnp.asarray(cfg.dt, dtype=rates.dtype)
    lo = jnp.asarray(cfg.state_min, dtype=rates.dtype)
    hi = jnp.asarray(cfg.state_max, dtype=rates.dtype)

    def step(state, _):
        # Classic RK4. The clamp after each stage keeps a diverging trajectory finite: an unclamped
        # LV run at wildly out-of-prior rates overflows to inf within a few steps and every
        # downstream gradient becomes NaN.
        k1 = _rhs(state, rates)
        k2 = _rhs(jnp.clip(state + 0.5 * dt * k1, lo, hi), rates)
        k3 = _rhs(jnp.clip(state + 0.5 * dt * k2, lo, hi), rates)
        k4 = _rhs(jnp.clip(state + dt * k3, lo, hi), rates)
        nxt = jnp.clip(state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), lo, hi)
        return nxt, nxt

    init = jnp.stack([jnp.asarray(cfg.x0, dtype=rates.dtype), jnp.asarray(cfg.y0, dtype=rates.dtype)])
    _, trajectory = jax.lax.scan(step, init, None, length=cfg.n_steps)  # (n_steps, 2)
    # Observation times are strides into the solver grid (t = stride*dt, 2*stride*dt, ...).
    idx = (jnp.arange(cfg.n_obs) + 1) * cfg.obs_stride - 1
    return trajectory[idx]


#: ``(batch, 4) -> (batch, n_obs, 2)``.
lv_states_batch = jax.jit(jax.vmap(lv_states, in_axes=(0, None)), static_argnames="cfg")


def add_observation_noise(states: jnp.ndarray, key: jax.Array, cfg: LVConfig) -> jnp.ndarray:
    """Apply multiplicative lognormal noise: ``states * exp(obs_sigma * eps)``."""
    eps = jax.random.normal(key, states.shape, dtype=states.dtype)
    return states * jnp.exp(cfg.obs_sigma * eps)


# --------------------------------------------------------------------------------------------- #
# Prior, likelihood, posterior
# --------------------------------------------------------------------------------------------- #

_LOG_2PI = float(np.log(2.0 * np.pi))


def _prior_moments(theta_log: jnp.ndarray, cfg: LVConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dtype = theta_log.dtype
    return (
        jnp.asarray(cfg.prior_mean, dtype=dtype),
        jnp.asarray(cfg.prior_std, dtype=dtype),
    )


@functools.partial(jax.jit, static_argnames="cfg")
def log_prior(theta_log: jnp.ndarray, cfg: LVConfig) -> jnp.ndarray:
    """Independent Normal log-prior on the log-rates. Scalar output for ``theta_log`` of shape ``(4,)``."""
    mean, std = _prior_moments(theta_log, cfg)
    z = (theta_log - mean) / std
    return jnp.sum(-0.5 * jnp.square(z) - jnp.log(std) - 0.5 * _LOG_2PI)


@functools.partial(jax.jit, static_argnames="cfg")
def log_likelihood(theta_log: jnp.ndarray, x_obs: jnp.ndarray, cfg: LVConfig) -> jnp.ndarray:
    """Lognormal observation log-likelihood ``log p(x_obs | theta_log)``.

    Parameters
    ----------
    theta_log : array of shape ``(4,)``
    x_obs : array of shape ``(n_obs, 2)``
        Observed abundances in **physical units** (strictly positive).

    Returns
    -------
    scalar
        Includes the full normalizing constant, so this is a proper log-density and can be fed to
        MCMC directly. The ``-log x_obs`` Jacobian term is constant in ``theta_log`` and so does
        not affect the guidance gradient.
    """
    states = lv_states(theta_log, cfg)
    sigma = jnp.asarray(cfg.obs_sigma, dtype=states.dtype)
    log_obs = jnp.log(jnp.maximum(x_obs, cfg.state_min))
    z = (log_obs - jnp.log(states)) / sigma
    return jnp.sum(-0.5 * jnp.square(z) - jnp.log(sigma) - 0.5 * _LOG_2PI - log_obs)


@functools.partial(jax.jit, static_argnames="cfg")
def log_posterior(theta_log: jnp.ndarray, x_obs: jnp.ndarray, cfg: LVConfig) -> jnp.ndarray:
    """Unnormalized log-posterior — the target for the MALA reference sampler."""
    return log_prior(theta_log, cfg) + log_likelihood(theta_log, x_obs, cfg)


#: ``grad_log_likelihood(theta_log, x_obs, cfg) -> (4,)``: the guidance term.
grad_log_likelihood = jax.jit(jax.grad(log_likelihood, argnums=0), static_argnames="cfg")

#: ``(batch, 4), (n_obs, 2) -> (batch, 4)``: the guidance term for a whole batch of particles
#: against a single observation (which is how the guided sampler is driven — one observation per
#: ``sample`` call, so every particle shares ``x_obs`` and no per-row alignment is needed).
grad_log_likelihood_batch = jax.jit(
    jax.vmap(grad_log_likelihood, in_axes=(0, None, None)), static_argnames="cfg"
)

#: ``(batch, 4), (n_obs, 2) -> (batch,)``.
log_likelihood_batch = jax.jit(
    jax.vmap(log_likelihood, in_axes=(0, None, None)), static_argnames="cfg"
)
log_posterior_batch = jax.jit(
    jax.vmap(log_posterior, in_axes=(0, None, None)), static_argnames="cfg"
)


def soft_clip_theta(theta_log: jnp.ndarray, cfg: LVConfig, n_std: float = 4.0) -> jnp.ndarray:
    """Smoothly squash log-parameters into ``mean +- n_std*std`` of the prior.

    Guidance evaluates the forward model at the *denoised estimate* ``x_hat_0``, which early in the
    reverse diffusion is essentially prior-free noise. This ``tanh`` squash keeps the value bounded
    (so the ODE stays sane) while remaining smooth — unlike a hard clip, which puts a kink in the
    vector field.

    Note the derivative *decays* with distance and underflows to exactly 0 in float32 once the input
    is more than ~10 half-widths outside the box (``tanh`` saturates). That is intended: absurd
    denoised estimates then contribute no guidance at all. Within a few prior standard deviations —
    the regime that matters — the map is close to the identity with an O(1) derivative.
    """
    mean, std = _prior_moments(theta_log, cfg)
    half_width = n_std * std
    return mean + half_width * jnp.tanh((theta_log - mean) / half_width)


# --------------------------------------------------------------------------------------------- #
# Prior sampling (NumPy side, so the pipeline's reproducibility contract is preserved)
# --------------------------------------------------------------------------------------------- #


def sample_prior(n: int, rng: np.random.Generator, cfg: LVConfig) -> np.ndarray:
    """Draw ``(n, 4)`` log-parameters from the prior using the pipeline-supplied NumPy generator.

    The pipeline owns the RNG (``pipeline.io.run_chunked`` seeds one per chunk), so prior draws stay
    bit-reproducible and chunk-resumable. Only the observation noise needs a JAX key.
    """
    mean = np.asarray(cfg.prior_mean, dtype=np.float64)
    std = np.asarray(cfg.prior_std, dtype=np.float64)
    return rng.normal(loc=mean, scale=std, size=(n, 4))


def key_from_rng(rng: np.random.Generator) -> jax.Array:
    """Derive a JAX PRNG key from the pipeline's NumPy generator.

    Keeps the JAX-side stochasticity (observation noise) tied to ``cfg.seed`` and to the chunk
    seeding in ``pipeline.io``, instead of introducing an independent global key.
    """
    return jax.random.key(int(rng.integers(0, 2**63 - 1)))


# --------------------------------------------------------------------------------------------- #
# Self-test (no BayesFlow / Keras needed)
# --------------------------------------------------------------------------------------------- #


def _selftest() -> None:
    """Sanity-check the solver, the gradient, and the likelihood optimum. Raises on failure."""
    cfg = make_config()
    theta = jnp.asarray(cfg.prior_mean)

    states = lv_states(theta, cfg)
    assert states.shape == (cfg.n_obs, 2), states.shape
    assert bool(jnp.all(jnp.isfinite(states))) and bool(jnp.all(states > 0))

    # Noise-free data => the truth is an exact stationary point: the standardized residual z is
    # identically zero, so d(-0.5*z^2)/dtheta = -z * dz/dtheta vanishes bit-exactly.
    x_obs = states
    grad_at_truth = grad_log_likelihood(theta, x_obs, cfg)
    assert grad_at_truth.shape == (4,)
    assert float(jnp.max(jnp.abs(grad_at_truth))) == 0.0, f"grad at noise-free truth: {grad_at_truth}"

    # Finite-difference check *off* the optimum, where the gradient is O(100) and the difference
    # quotient is well above float32 noise. (Checking FD at the optimum measures only rounding.)
    probe = theta + 0.05
    grad = grad_log_likelihood(probe, x_obs, cfg)
    assert float(jnp.max(jnp.abs(grad))) > 1.0, f"gradient suspiciously small off-optimum: {grad}"
    eps = 1e-2
    for i in range(4):
        plus = probe.at[i].add(eps)
        minus = probe.at[i].add(-eps)
        fd = (log_likelihood(plus, x_obs, cfg) - log_likelihood(minus, x_obs, cfg)) / (2 * eps)
        rel = abs(float(fd) - float(grad[i])) / max(1.0, abs(float(fd)))
        assert rel < 5e-2, f"grad[{i}] mismatch: analytic {grad[i]} vs fd {fd}"

    # Extreme out-of-prior parameters must not produce NaN/inf gradients (the clamp doing its job).
    wild = jnp.asarray([8.0, 8.0, 8.0, 8.0])
    gw = grad_log_likelihood(wild, x_obs, cfg)
    assert bool(jnp.all(jnp.isfinite(gw))), f"non-finite gradient at wild theta: {gw}"

    # Batched paths agree with the single-example paths.
    batch = jnp.stack([probe, probe + 0.1])
    assert lv_states_batch(batch, cfg).shape == (2, cfg.n_obs, 2)
    assert jnp.allclose(grad_log_likelihood_batch(batch, x_obs, cfg)[0], grad, rtol=1e-4)

    print(f"lv_jax self-test OK  (n_steps={cfg.n_steps}, obs_times={cfg.obs_times})")


if __name__ == "__main__":
    _selftest()
