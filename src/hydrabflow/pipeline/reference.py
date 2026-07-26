"""Exact reference posterior by MALA, for judging guided samples against ground truth.

Simulation-based calibration tells you whether a posterior is *self-consistent* across many
observations; it cannot tell you whether the posterior for **one** observation is right. Since the
guidance study rests on a simulator with a tractable, differentiable likelihood, we can do better:
run gradient-based MCMC on the exact log-posterior and compare per-observation.

Sampler
-------
MALA (Metropolis-adjusted Langevin) — a gradient-preconditioned random walk with an exact Metropolis
correction, so the stationary distribution is the true posterior regardless of step size:

    proposal:  y = x + 0.5 * eps^2 * grad log p(x) + eps * xi,   xi ~ N(0, I)
    accept with the usual asymmetric-proposal Hastings ratio.

MALA rather than HMC deliberately: with only 4 unconstrained parameters there is no integrator to get
subtly wrong, which matters when this is the thing everything else is measured against.

**Whitening is not optional here.** Run naively, this sampler fails badly on the LV posterior: with
20 data points at 10% noise the posterior is tight and strongly correlated, the adapted step size
collapses (~1e-3) to satisfy the tightest direction, and mixing along the loosest direction dies
(measured ``r_hat ~ 74``, ESS ~35 out of 1500 draws). An isotropic proposal simply cannot serve an
anisotropic target.

So :func:`reference_posterior` preconditions first:

1. find the MAP by L-BFGS on the exact log-posterior (using the JAX gradient);
2. take the exact 4x4 Hessian there and use the Laplace covariance ``(-H)^-1`` as a whitening map
   ``L`` (Cholesky), falling back to a diagonal or identity map if it is not positive definite;
3. run MALA on the *whitened* density ``y -> log p(mu + L y)``, which is approximately standard
   normal, so a step size near 1 mixes well;
4. transform the draws back with ``x = mu + L y``.

Whitening changes only the proposal, never the target, so the draws remain exact samples from the
posterior. Diagnostics (``r_hat``, ``ess``, ``accept_rate``) are returned alongside them, so a badly
mixed reference is *visible* rather than silently treated as truth.

Parameters live in the simulator's unconstrained (log) space, so no change of variables is needed.

Status / intended replacement
----------------------------
This module is **opt-in and not yet validated** (``eval.n_reference_obs`` defaults to 0, so the
guided evaluation skips it). Whitening fixed the mixing (``r_hat`` 1.0006, ESS ~6900/16000), but the
MAP search remains the weak link: a single L-BFGS start was measured getting stuck at
``log p = -764`` on an observation whose truth scores ``-33``, and the multi-start fix is slow.

The intended replacement is **NUTS via numpyro** (or blackjax) rather than more hand-rolled MCMC —
mass-matrix adaptation and a proper trajectory integrator handle this anisotropic posterior without
the MAP-finding step, and step-size/mass adaptation is battle-tested. Neither package is currently a
dependency. Until then, treat any reference-based number as provisional.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

import numpy as np

from hydrabflow.utils.logging import get_logger

log = get_logger(__name__)

#: Defaults for ``eval.reference``.
DEFAULTS: Dict[str, Any] = {
    "num_chains": 8,
    "num_warmup": 2000,
    "num_samples": 2000,
    "thin": 1,
    # Step size is in WHITENED units, where the target is ~standard normal, so O(1) is right.
    "step_size": 0.5,
    "target_accept": 0.6,
    "adapt_rate": 0.05,
    # Chain-initialization dispersion in whitened units (1.0 = draw from the Laplace approximation).
    "init_scale": 1.0,
    # Set false to sample the raw parameterization (diagnostic only — expect terrible mixing).
    "whiten": True,
    # L-BFGS restarts for the MAP search. A single start was measured getting stuck far from the
    # mode on LV observations, silently yielding a confident but wrong reference.
    "map_restarts": 32,
}


def resolve_settings(overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Merge ``eval.reference`` over :data:`DEFAULTS`, rejecting unknown keys."""
    from omegaconf import OmegaConf

    settings = dict(DEFAULTS)
    if overrides:
        if OmegaConf.is_config(overrides):
            overrides = OmegaConf.to_container(overrides, resolve=True)
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise ValueError(
                f"unknown eval.reference key(s) {sorted(unknown)}; valid: {sorted(DEFAULTS)}"
            )
        settings.update(overrides)
    for key in ("num_chains", "num_warmup", "num_samples", "thin", "map_restarts"):
        settings[key] = int(settings[key])
    for key in ("step_size", "target_accept", "adapt_rate", "init_scale"):
        settings[key] = float(settings[key])
    settings["whiten"] = bool(settings["whiten"])
    if settings["num_samples"] <= 0 or settings["num_chains"] <= 0:
        raise ValueError("eval.reference.num_samples and num_chains must be positive")
    return settings


def mala_sample(
    log_density: Callable,
    initial_state: np.ndarray,
    settings: Mapping[str, Any],
    seed: int = 0,
) -> Dict[str, Any]:
    """Run MALA on ``log_density``.

    Parameters
    ----------
    log_density : callable
        ``(theta,) -> scalar`` unnormalized log-density for a **single** parameter vector, traceable
        by JAX. Vectorized over chains internally.
    initial_state : array of shape ``(num_chains, dim)``
    settings : mapping
        See :data:`DEFAULTS`.

    Returns
    -------
    dict
        ``samples`` ``(num_chains, num_kept, dim)``, plus ``accept_rate``, ``step_size``, ``r_hat``
        and ``ess`` (per dimension).
    """
    import jax
    import jax.numpy as jnp

    num_warmup = settings["num_warmup"]
    num_samples = settings["num_samples"]
    thin = max(1, settings["thin"])
    target_accept = settings["target_accept"]
    adapt_rate = settings["adapt_rate"]

    value_and_grad = jax.vmap(jax.value_and_grad(log_density))

    def step(carry, key):
        x, logp, grad, log_eps, adapting = carry
        eps = jnp.exp(log_eps)
        key_prop, key_acc = jax.random.split(key)

        half_eps2 = 0.5 * jnp.square(eps)
        noise = jax.random.normal(key_prop, x.shape, dtype=x.dtype)
        y = x + half_eps2 * grad + eps * noise
        logp_y, grad_y = value_and_grad(y)

        # Asymmetric-proposal (Hastings) correction: log q(x|y) - log q(y|x).
        def log_q(dest, src, src_grad):
            mean = src + half_eps2 * src_grad
            return -jnp.sum(jnp.square(dest - mean), axis=-1) / (2.0 * jnp.square(eps))

        log_ratio = (logp_y - logp) + log_q(x, y, grad_y) - log_q(y, x, grad)
        log_ratio = jnp.where(jnp.isfinite(log_ratio), log_ratio, -jnp.inf)
        accept = jnp.log(jax.random.uniform(key_acc, logp.shape, dtype=x.dtype)) < log_ratio

        keep = accept[:, None]
        x = jnp.where(keep, y, x)
        logp = jnp.where(accept, logp_y, logp)
        grad = jnp.where(keep, grad_y, grad)

        # Robbins-Monro adaptation of a single shared step size, warmup only.
        rate = jnp.mean(accept.astype(x.dtype))
        log_eps = log_eps + adapting * adapt_rate * (rate - target_accept)
        return (x, logp, grad, log_eps, adapting), (x, rate)

    x0 = jnp.asarray(initial_state)
    logp0, grad0 = value_and_grad(x0)
    if not bool(jnp.all(jnp.isfinite(logp0))):
        raise ValueError("MALA initial state has non-finite log-density; check the initialization")

    key = jax.random.PRNGKey(int(seed))
    warm_keys, sample_keys = jax.random.split(key)

    carry = (x0, logp0, grad0, jnp.log(jnp.asarray(settings["step_size"], dtype=x0.dtype)), 1.0)
    if num_warmup > 0:
        carry, (_, warm_rates) = jax.lax.scan(
            step, carry, jax.random.split(warm_keys, num_warmup)
        )
        log.info(
            "MALA warmup: accept %.3f -> %.3f, step size %.4g",
            float(warm_rates[0]),
            float(jnp.mean(warm_rates[-100:])),
            float(jnp.exp(carry[3])),
        )

    # Freeze the step size for the sampling phase, so the chain is a valid Markov chain.
    carry = (*carry[:4], 0.0)
    carry, (chain, rates) = jax.lax.scan(step, carry, jax.random.split(sample_keys, num_samples))

    chain = jnp.swapaxes(chain, 0, 1)[:, ::thin]  # (chains, kept, dim)
    samples = np.asarray(chain)
    return {
        "samples": samples,
        "accept_rate": float(jnp.mean(rates)),
        "step_size": float(jnp.exp(carry[3])),
        "r_hat": _r_hat(samples),
        "ess": _ess(samples),
    }


def _r_hat(chain: np.ndarray) -> np.ndarray:
    """Split-free rank-1 Gelman-Rubin statistic per dimension. ``chain`` is ``(chains, draws, dim)``."""
    n_chains, n_draws, _ = chain.shape
    if n_chains < 2 or n_draws < 2:
        return np.full(chain.shape[-1], np.nan)
    chain_means = chain.mean(axis=1)
    within = chain.var(axis=1, ddof=1).mean(axis=0)
    between = chain_means.var(axis=0, ddof=1)
    var_hat = (n_draws - 1) / n_draws * within + between
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_hat / within)


def _ess(chain: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """Effective sample size per dimension from the summed positive autocorrelations."""
    n_chains, n_draws, dim = chain.shape
    if n_draws < 4:
        return np.full(dim, np.nan)
    ess = np.zeros(dim)
    lag_cap = min(max_lag, n_draws // 2)
    for d in range(dim):
        rho_sum = 0.0
        centred = chain[:, :, d] - chain[:, :, d].mean(axis=1, keepdims=True)
        var = (centred**2).mean()
        if var <= 0:
            ess[d] = np.nan
            continue
        for lag in range(1, lag_cap):
            rho = (centred[:, :-lag] * centred[:, lag:]).mean() / var
            if rho <= 0.05:  # truncate at the first (near-)zero autocorrelation
                break
            rho_sum += rho
        ess[d] = n_chains * n_draws / (1.0 + 2.0 * rho_sum)
    return ess


def laplace_preconditioner(log_density: Callable, starts: np.ndarray) -> Dict[str, Any]:
    """MAP + Laplace covariance of ``log_density``, as a whitening map.

    ``starts`` is ``(n_starts, dim)``: L-BFGS is run from **every** row and the best optimum kept.
    Multi-start is not a luxury here — a single start from the prior mean was measured landing at
    ``log p = -764`` on an LV observation whose true parameters score ``-33``, i.e. stuck in a flat
    region far from the mode. That silently produced a confidently wrong reference posterior
    (``r_hat = 1.0006`` at 14 sigma from the truth).

    Returns ``{"mu", "L", "cov", "kind", "logp_map", "n_starts_converged"}``, where ``x = mu + L @ y``
    maps whitened coordinates back to parameter space. ``kind`` records whether the full Cholesky, a
    diagonal fallback (non-PD Hessian) or the identity (failed optimization) is in use, so a degraded
    preconditioner is visible in the saved metadata rather than silent.
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    starts = np.atleast_2d(np.asarray(starts, dtype=np.float64))
    dim = starts.shape[1]
    # Jit value-and-grad ONCE. Without this, `map_restarts` L-BFGS runs re-enter the JAX dispatch
    # path thousands of times and the MAP search dominates the whole stage's runtime.
    value_and_grad = jax.jit(jax.value_and_grad(log_density))

    # L-BFGS on the negative log-posterior. float64 on the numpy side keeps the optimizer happy even
    # though the density itself is evaluated in float32.
    def neg_and_grad(theta_np):
        value, grad = value_and_grad(jnp.asarray(theta_np, dtype=jnp.float32))
        value = -float(value)
        grad = np.nan_to_num(-np.asarray(grad, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return (value if np.isfinite(value) else 1e30), grad

    def neg(theta_np):
        return neg_and_grad(theta_np)[0]

    kind = "cholesky"
    best_mu, best_value, n_ok = None, np.inf, 0
    for start in starts:
        try:
            opt = minimize(neg_and_grad, start, jac=True, method="L-BFGS-B")
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("MAP start failed: %s", exc)
            continue
        if not np.all(np.isfinite(opt.x)):
            continue
        n_ok += 1
        value = neg(opt.x)
        if value < best_value:
            best_mu, best_value = np.asarray(opt.x, dtype=np.float64), value

    if best_mu is None:  # pragma: no cover - defensive
        log.warning("Every MAP optimization failed; falling back to the first start point.")
        mu, kind, best_value = starts[0], "identity", neg(starts[0])
    else:
        mu = best_mu
    logp_map = -float(best_value)

    cov = np.eye(dim)
    if kind != "identity":
        hess = np.asarray(
            jax.hessian(log_density)(jnp.asarray(mu, dtype=jnp.float32)), dtype=np.float64
        )
        precision = -0.5 * (hess + hess.T)  # symmetrize; -H is the Laplace precision
        eigvals = np.linalg.eigvalsh(precision)
        if np.all(eigvals > 1e-8):
            cov = np.linalg.inv(precision)
        else:
            # Not positive definite (saddle / flat direction): keep only the diagonal curvature.
            diag = np.diag(precision)
            if np.all(diag > 1e-8):
                cov, kind = np.diag(1.0 / diag), "diagonal"
                log.warning("Laplace Hessian not positive definite; using diagonal curvature.")
            else:
                cov, kind = np.eye(dim), "identity"
                log.warning("Laplace Hessian unusable; sampling without preconditioning.")

    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        chol, kind = np.eye(dim), "identity"
        log.warning("Laplace covariance not factorizable; sampling without preconditioning.")

    return {
        "mu": mu,
        "L": chol,
        "cov": cov,
        "kind": kind,
        "logp_map": logp_map,
        "n_starts_converged": n_ok,
    }


def reference_posterior(
    simulator,
    x_obs: np.ndarray,
    settings: Mapping[str, Any],
    seed: int = 0,
) -> Dict[str, Any]:
    """MALA draws from ``p(theta | x_obs)`` for one observation, using the simulator's own densities.

    ``x_obs`` is in physical units (as written by ``simulate``). The chain runs in whitened
    coordinates derived from a Laplace approximation at the MAP (see the module docstring): without
    that, the LV posterior's anisotropy makes plain MALA useless. Whitening affects the proposal
    only, so the returned draws are exact posterior samples.
    """
    import jax.numpy as jnp

    log_likelihood_batch = simulator.jax_log_likelihood(x_obs)
    log_prior_batch = simulator.jax_log_prior()

    def log_density(theta):
        one = theta[None, :]
        return (log_prior_batch(one) + log_likelihood_batch(one))[0]

    names = list(simulator.parameter_names)
    dim = len(names)
    rng = np.random.default_rng(seed)

    def _draw(n: int) -> np.ndarray:
        params = simulator.sample_prior(n, rng)
        return np.concatenate([np.asarray(params[k]).reshape(n, 1) for k in names], axis=1)

    if not settings.get("whiten", True):
        initial = jnp.asarray(_draw(settings["num_chains"]), dtype=jnp.float32)
        result = mala_sample(log_density, initial, settings, seed=seed)
        result["preconditioner"] = {"kind": "none"}
        result["flat_samples"] = result["samples"].reshape(-1, dim)
        _log_reference(result)
        return result

    n_starts = int(settings["map_restarts"])
    starts = np.vstack([_draw(256).mean(axis=0)[None, :], _draw(max(0, n_starts - 1))])
    pre = laplace_preconditioner(log_density, starts)
    mu = jnp.asarray(pre["mu"], dtype=jnp.float32)
    chol = jnp.asarray(pre["L"], dtype=jnp.float32)

    def whitened_density(y):
        return log_density(mu + chol @ y)

    # Initialize from PRIOR draws mapped into whitened coordinates, NOT from the Laplace
    # approximation. Starting every chain at the MAP makes r_hat uninformative by construction:
    # the chains cannot disagree, so a mode-trapped or mis-located reference reports r_hat ~ 1.
    # Dispersed prior starts make r_hat an honest multimodality/convergence detector; the whitening
    # is still doing its job of making the step size usable.
    chol_inv = np.linalg.inv(np.asarray(pre["L"]))
    prior_start = _draw(settings["num_chains"])
    initial = jnp.asarray(
        settings["init_scale"] * (prior_start - np.asarray(pre["mu"])) @ chol_inv.T,
        dtype=jnp.float32,
    )
    result = mala_sample(whitened_density, initial, settings, seed=seed)

    # Back to parameter space: x = mu + L y.
    samples_y = result["samples"]
    result["samples"] = np.asarray(pre["mu"]) + samples_y @ np.asarray(pre["L"]).T
    result["flat_samples"] = result["samples"].reshape(-1, dim)
    # r_hat / ESS are transform-equivariant in the sense that matters (a fixed linear map), but
    # recompute on the parameter-space draws so the reported numbers describe what is returned.
    result["r_hat"] = _r_hat(result["samples"])
    result["ess"] = _ess(result["samples"])
    result["preconditioner"] = {
        "kind": pre["kind"],
        "map": pre["mu"].tolist(),
        "logp_map": pre["logp_map"],
        "n_starts_converged": pre["n_starts_converged"],
        "laplace_std": np.sqrt(np.diag(pre["cov"])).tolist(),
    }
    _log_reference(result)
    return result


def _log_reference(result: Dict[str, Any]) -> None:
    log.info(
        "MALA reference: accept %.3f, step %.4g, precond %s, max r_hat %.4f, min ESS %.0f",
        result["accept_rate"],
        result["step_size"],
        result.get("preconditioner", {}).get("kind", "?"),
        float(np.nanmax(result["r_hat"])),
        float(np.nanmin(result["ess"])),
    )
