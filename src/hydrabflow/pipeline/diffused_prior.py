"""Exact (or semi-analytic) score of the DIFFUSED prior, for compositional score sampling.

Compositional sampling combines per-item posterior scores with ``(1 - n)`` times the score of the
prior *at the same diffusion time*. Under the forward process ``x_t = alpha_t x_0 + sigma_t eps``
the time-t prior is ``p_t = p_0 * N(0, sigma_t^2)`` (after the ``alpha_t`` rescaling), so the score
that belongs in the compositional sum is ``grad log p_t`` — not the clean prior score damped by an
ad-hoc ``(1 - t)``. This module computes it for priors that factorize into:

(a) Gaussian parameters ``N(m, s^2)``: exact, ``-(x - alpha m) / (alpha^2 s^2 + sigma^2)``.
(b) Uniform parameters ``U[a, b]``: ``p_t(x) ∝ Phi((x - alpha a)/sigma) - Phi((x - alpha b)/sigma)``,
    logged stably with ``log_ndtr`` and a log-diff-exp, differentiated with ``jax.grad``.
(c) A uniform parent ``l ~ U[a, b]`` with a ``moster_conditional`` child
    ``l* | l ~ N(mu_moster(l), s^2)``: the child convolves exactly, the parent integral is done by
    Gauss-Legendre quadrature on a LOCAL grid (+-``half_width`` sigma_t/alpha_t around
    ``x1/alpha_t``, clipped to [a, b]) so it resolves the narrow parent Gaussian even at the
    sampler's smallest sigma_t (~1e-4 for the EDM schedule). The grid bounds sit under
    ``stop_gradient``; the integrand is negligible at an interior bound and the prior bounds are
    fixed, so the derivative of the quadrature sum is the integral of the derivative.

Everything works in the NETWORK's standardized space ``z = (theta - mean) / std``: every prior is
transformed accordingly (means, widths, bounds; the Moster conditional mean becomes
``(mu_moster(l) - mean_child) / std_child``). ``mean=0, std=1`` gives theta space.

Only INFERRED parameters enter: nuisance (marginalized) parameters are absent from
``simulator.prior_spec_global`` and from the inference variables, so they never reach this score.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from hydrabflow.simulators.stream_common import moster_params, mu_moster


def _blocks(prior_spec: Mapping[str, Mapping], names: Sequence[str], mean, std):
    """Split the prior into (gaussian, uniform, moster) blocks, in standardized coordinates."""
    idx = {n: i for i, n in enumerate(names)}
    parents = {
        str(s["condition_on"]): n for n, s in prior_spec.items()
        if n in idx and s["type"] == "moster_conditional"
    }
    gauss, unif, cond = [], [], []
    for n in names:
        spec = prior_spec[n]
        kind = spec["type"]
        p = [float(v) for v in spec["prior_parameters"]]
        i = idx[n]
        mu, sd = float(mean[i]), float(std[i])
        if kind == "normal":
            gauss.append((i, (p[0] - mu) / sd, p[1] / sd))
        elif kind == "uniform":
            if n not in parents:
                unif.append((i, (p[0] - mu) / sd, (p[1] - mu) / sd))
        elif kind == "moster_conditional":
            parent = str(spec["condition_on"])
            if parent not in idx:
                raise ValueError(f"{n} conditions on {parent!r}, which is not an inferred parameter")
            pspec = prior_spec[parent]
            if pspec["type"] != "uniform":
                raise NotImplementedError(f"moster_conditional parent {parent!r} must be uniform")
            ip = idx[parent]
            a, b = (float(v) for v in pspec["prior_parameters"])
            cond.append(dict(
                i_parent=ip, i_child=i, a=a, b=b,
                mu_p=float(mean[ip]), sd_p=float(std[ip]), mu_c=mu, sd_c=sd,
                scatter=p[0], moster=moster_params(spec),
            ))
        else:
            raise NotImplementedError(
                f"diffused prior score: unsupported prior type {kind!r} for {n!r} "
                "(supported: normal, uniform, moster_conditional)"
            )
    return gauss, unif, cond


def make_log_prob_t(
    prior_spec: Mapping[str, Mapping],
    names: Sequence[str],
    mean=None,
    std=None,
    n_quad: int = 128,
    half_width: float = 10.0,
) -> Callable:
    """``log_prob_t(theta, alpha_t, sigma_t) -> (B,)``: the NORMALIZED log density of the diffused
    prior in standardized coordinates. ``theta`` is ``(B, D)`` in the order of ``names``;
    ``alpha_t``/``sigma_t`` are scalars or per-row ``(B,)``/``(B, 1)``. Pure JAX, jit-able."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import log_ndtr, logsumexp

    D = len(names)
    mean = np.zeros(D) if mean is None else np.asarray(mean, dtype=float).reshape(D)
    std = np.ones(D) if std is None else np.asarray(std, dtype=float).reshape(D)
    gauss, unif, cond = _blocks(prior_spec, names, mean, std)
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))  # on [-1, 1]
    half_log_2pi = 0.5 * np.log(2.0 * np.pi)

    def log_normal(x, loc, var):
        return -0.5 * (x - loc) ** 2 / var - 0.5 * jnp.log(var) - half_log_2pi

    def log1mexp(x):  # log(1 - exp(x)) for x < 0, both branches kept finite for grad
        x = jnp.minimum(x, -1e-300)
        small = x > -np.log(2.0)
        xs = jnp.where(small, x, -1.0)
        xl = jnp.where(small, -1.0, x)
        return jnp.where(small, jnp.log(-jnp.expm1(xs)), jnp.log1p(-jnp.exp(xl)))

    def log_uniform_t(x, a, b, alpha, sigma):
        u1 = (x - alpha * a) / sigma
        u2 = (x - alpha * b) / sigma  # u2 < u1
        upper = (u1 + u2) > 0  # above the midpoint: use the mirrored tail Phi(-u2) - Phi(-u1)
        hi = jnp.where(upper, -u2, u1)
        lo = jnp.where(upper, -u1, u2)
        lhi = log_ndtr(hi)
        return lhi + log1mexp(log_ndtr(lo) - lhi) - jnp.log(alpha * (b - a))

    def log_moster_t(x1, x2, c, alpha, sigma):
        a, b = c["a"], c["b"]
        mu_p, sd_p, mu_c, sd_c = c["mu_p"], c["sd_p"], c["mu_c"], c["sd_c"]
        # local grid in raw l: centre on the l implied by x1/alpha, +-half_width parent sigmas
        l_c = jnp.clip(mu_p + sd_p * x1 / alpha, a, b)
        w = half_width * sd_p * sigma / alpha
        lo = jax.lax.stop_gradient(jnp.maximum(a, l_c - w))
        hi = jax.lax.stop_gradient(jnp.minimum(b, l_c + w))
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        off = half[:, None] * nodes[None, :]  # (B, Q) offsets: keeps float32 precision at tiny sigma
        ell = mid[:, None] + off
        log_w = jnp.log(half)[:, None] + np.log(weights)[None, :]
        z1 = ((mid - mu_p) / sd_p)[:, None] + off / sd_p
        m2 = (mu_moster(ell, xp=jnp, **c["moster"]) - mu_c) / sd_c
        s2 = c["scatter"] / sd_c
        a_ = alpha[:, None]
        s_ = sigma[:, None]
        lg = (
            log_normal(x1[:, None], a_ * z1, s_**2)
            + log_normal(x2[:, None], a_ * m2, a_**2 * s2**2 + s_**2)
        )
        return logsumexp(lg + log_w, axis=1) - np.log(b - a)

    def log_prob_t(theta, alpha_t, sigma_t):
        theta = jnp.asarray(theta)
        B = theta.shape[0]
        alpha = jnp.broadcast_to(jnp.reshape(jnp.asarray(alpha_t, theta.dtype), (-1,)), (B,))
        sigma = jnp.broadcast_to(jnp.reshape(jnp.asarray(sigma_t, theta.dtype), (-1,)), (B,))
        out = jnp.zeros((B,), theta.dtype)
        for i, m, s in gauss:
            out = out + log_normal(theta[:, i], alpha * m, alpha**2 * s**2 + sigma**2)
        for i, a, b in unif:
            out = out + log_uniform_t(theta[:, i], a, b, alpha, sigma)
        for c in cond:
            out = out + log_moster_t(theta[:, c["i_parent"]], theta[:, c["i_child"]], c, alpha, sigma)
        return out

    return log_prob_t


def make_prior_score_t(
    prior_spec: Mapping[str, Mapping],
    names: Sequence[str],
    mean=None,
    std=None,
    n_quad: int = 128,
    half_width: float = 10.0,
) -> Callable:
    """``prior_score_t(theta, alpha_t, sigma_t) -> (B, D)``: ``grad_theta log p_t`` in standardized
    coordinates (see the module docstring). The Gaussian block is returned in closed form, the
    rest through ``jax.grad`` of :func:`make_log_prob_t`. Pure JAX, jit-able."""
    import jax
    import jax.numpy as jnp

    D = len(names)
    mean_ = np.zeros(D) if mean is None else np.asarray(mean, dtype=float).reshape(D)
    std_ = np.ones(D) if std is None else np.asarray(std, dtype=float).reshape(D)
    gauss, _, _ = _blocks(prior_spec, names, mean_, std_)
    g_idx = [i for i, _, _ in gauss]
    rest = {n: prior_spec[n] for i, n in enumerate(names) if i not in g_idx}
    rest_names = [n for i, n in enumerate(names) if i not in g_idx]
    r_idx = [i for i in range(D) if i not in g_idx]
    log_rest = (
        make_log_prob_t(rest, rest_names, mean_[r_idx], std_[r_idx], n_quad, half_width)
        if rest_names else None
    )
    g_i = np.asarray(g_idx, dtype=int)
    g_m = np.asarray([m for _, m, _ in gauss], dtype=float)
    g_s = np.asarray([s for _, _, s in gauss], dtype=float)
    r_i = np.asarray(r_idx, dtype=int)

    def prior_score_t(theta, alpha_t, sigma_t):
        theta = jnp.asarray(theta)
        B = theta.shape[0]
        alpha = jnp.broadcast_to(jnp.reshape(jnp.asarray(alpha_t, theta.dtype), (-1, 1)), (B, 1))
        sigma = jnp.broadcast_to(jnp.reshape(jnp.asarray(sigma_t, theta.dtype), (-1, 1)), (B, 1))
        out = jnp.zeros_like(theta)
        if len(g_i):
            xg = theta[:, g_i]
            out = out.at[:, g_i].set(-(xg - alpha * g_m) / (alpha**2 * g_s**2 + sigma**2))
        if log_rest is not None:
            grad = jax.grad(lambda x: jnp.sum(log_rest(x, alpha[:, 0], sigma[:, 0])))(theta[:, r_i])
            out = out.at[:, r_i].set(grad)
        return out

    return prior_score_t


def prior_score_from_diffused(prior_spec, param_order, approximator):
    """BayesFlow ``compute_prior_score`` callable using the exact diffused prior.

    ``bayesflow.approximators.helpers.compositional.build_prior_score_fn`` calls it with
    UN-standardized parameters and the diffusion ``time``, then multiplies the result by the
    ``inference_variables`` standardization std. So this standardizes theta with the network's own
    moving mean/std, maps ``time -> log_snr -> (alpha, sigma)`` through the inference network's
    noise schedule, evaluates the score in z-space and returns ``score_z / std`` per key. Because
    the signature names ``time``, BayesFlow applies no ``(1 - t)`` decay of its own (none is wanted).
    """
    import keras

    names = list(param_order)
    std_layer = None
    standardizer = getattr(approximator, "standardizer", None)
    if standardizer is not None and "inference_variables" in (standardizer.standardize or []):
        std_layer = standardizer.standardize_layers["inference_variables"]
    if std_layer is not None:
        mean = np.concatenate([keras.ops.convert_to_numpy(m) for m in std_layer.moving_mean])
        std = np.concatenate([
            keras.ops.convert_to_numpy(std_layer.moving_std(i))
            for i in range(len(std_layer.moving_mean))
        ])
    else:
        mean, std = np.zeros(len(names)), np.ones(len(names))
    if len(mean) != len(names):
        raise ValueError(f"standardizer width {len(mean)} != {len(names)} inference variables")
    return _diffused_callable(
        prior_spec, names, mean, std, approximator.inference_network.noise_schedule
    )


def _diffused_callable(prior_spec, names, mean, std, schedule):
    import jax.numpy as jnp

    score_z = make_prior_score_t(prior_spec, names, mean, std)
    mean_j = jnp.asarray(mean, dtype=jnp.float32)
    std_j = jnp.asarray(std, dtype=jnp.float32)

    def score(x, time=None):
        theta = jnp.concatenate([jnp.reshape(jnp.asarray(x[k]), (-1, 1)) for k in names], axis=1)
        shape = jnp.shape(x[names[0]])
        if time is None:  # clean prior: alpha=1, sigma -> 0 limit is not defined; use t = 0
            time = 0.0
        log_snr = schedule.get_log_snr(time, training=False)
        alpha_t, sigma_t = schedule.get_alpha_sigma(log_snr)
        z = (theta - mean_j) / std_j
        s = score_z(z, alpha_t, sigma_t) / std_j
        return {k: jnp.reshape(s[:, i], shape) for i, k in enumerate(names)}

    return score
