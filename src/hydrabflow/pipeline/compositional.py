"""Helpers for compositional (grouped) evaluation.

A compositional dataset stores ``n`` groups of ``m`` exchangeable members (e.g. m streams
sharing one potential): globals ``(n, 1)``, per-member arrays ``(n, m, ...)``, group-level
observables (like the rotation curve) ``(n, bins, 1)``. Augmentations operate on flat rows, so
evaluation flattens the member axis, augments once, and regroups before sampling.
"""

from __future__ import annotations

import math
from typing import Callable, Container, Dict, Mapping

import numpy as np


def composition_level(cfg) -> str:
    return str(getattr(getattr(cfg, "composition", None), "level", "none") or "none")


def prior_score_from_spec(
    prior_spec: Mapping[str, Mapping], log10_keys: Container[str] = ()
) -> Callable[[dict], dict]:
    """Score of the log prior for compositional sampling, from a prior-spec mapping.

    BayesFlow decides whether to apply its own ``(1 - t)`` time-decay by inspecting this
    callable's signature (``bayesflow.approximators.helpers.compositional``:
    ``prior_has_time = "time" in inspect.signature(compute_prior_score).parameters``): if the
    callable accepts a ``time`` argument, BayesFlow assumes the callable handles the decay
    itself and skips its own multiplication. Since this function's signature names a ``time``
    parameter, it MUST apply the ``(1 - t)`` factor explicitly here (matching the compositional
    formula ``(1-n)(1-t) grad log p(theta) + sum_i s_i``) — otherwise the raw, undecayed prior
    score is injected at every diffusion step (including near t=1, pure noise), which showed up
    as severe under-confidence for tight normal priors (e.g. z_Disk) while uniform priors were
    unaffected (their raw score is 0 regardless of decay).

    Uniform priors contribute zero score, normal priors ``-(x - mean) / std**2``. The function
    is traced inside the sampler's integration loop, so it uses backend ops, not NumPy.

    ``log10_keys`` lists parameters the ``log10_transform`` preprocessing step reparametrized
    (see ``preprocessing.steps.Log10Transform``): the prior in ``prior_spec`` (from the
    simulator's config) is defined in physical units ``x``, but the network — and hence this
    score, evaluated during compositional sampling — operates on ``y = log10(x)``. By the
    change-of-variables rule, for ``x = g(y) = 10**y`` (so ``g'(y) = g(y) * ln(10)``):

        d/dy log p_Y(y) = g'(y) * [d/dx log p_X(x)]|_{x=g(y)} + d/dy log(g'(y))
                         = x * ln(10) * score_X(x) + ln(10)

    (the ``+ln(10)`` term is the Jacobian's own contribution and is nonzero even for a uniform
    prior in ``x`` — omitting it silently biases the compositional posterior.)
    """
    from keras import ops

    ln10 = math.log(10.0)

    def score(x: Dict[str, np.ndarray], time=None) -> Dict[str, np.ndarray]:
        out = {}
        for key, arr in x.items():
            spec = prior_spec[key]
            is_log10 = key in log10_keys
            x_phys = 10.0**arr if is_log10 else arr

            if spec["type"] == "normal":
                mean, std = (float(p) for p in spec["prior_parameters"])
                base_score = -(x_phys - mean) / std**2
            else:  # uniform (flat inside the support)
                base_score = ops.zeros_like(arr)

            if is_log10:
                score_val = x_phys * ln10 * base_score + ln10
            else:
                score_val = base_score

            out[key] = score_val if time is None else (1.0 - time) * score_val
        return out

    return score


def prior_score_from_kde(
    samples_path: str,
    param_order,
    log10_keys: Container[str] = (),
    max_points: int = 4096,
    bandwidth: float = 0.0,
    seed: int = 0,
) -> Callable[[dict], dict]:
    """KDE compositional prior score, fit in the network's native (un-standardized, log10) space.

    Use this instead of :func:`prior_score_from_spec` when the training prior was truncated by
    ``vcirc_rejection``: the analytic spec is blind to the truncation boundary, a KDE of the
    actual (rejection-sampled) training draws is not.

    **Space contract (this is what the earlier physical-unit KDE got wrong).**
    ``bayesflow.approximators.helpers.compositional.build_prior_score_fn`` calls
    ``compute_prior_score`` on parameters that have already been *un-standardized* and passed
    through the *inverse* of the (zero-log-det-jac) adapter — i.e. the network's native space:
    ``log10(x)`` for the parameters the ``log10_transform`` preprocessing reparametrized and
    physical ``x`` for the rest. BayesFlow then multiplies the returned score by the
    standardization std itself. So the KDE must be fit on the training draws transformed to that
    SAME space (``log10`` on ``log10_keys``); then ``grad log p_KDE`` is directly the score in
    that space — no separate change-of-variables/Jacobian term is needed (unlike the analytic
    spec, whose ``+ln(10)`` term corrects the *closed-form* density, not a density fit in the
    transformed space). The old KDE was fit on the raw physical-unit npz arrays but evaluated on
    ``log10``-space parameters, so its gradient was wrong precisely for the large-magnitude
    ``log10_keys`` (rho, Sigma) — which were the worst-recovered parameters.

    Because the callable names a ``time`` parameter, BayesFlow expects it to apply the ``(1-t)``
    time-decay itself (same convention as :func:`prior_score_from_spec`); we do so here.

    A diagonal-bandwidth Gaussian KDE is used (per-dimension bandwidth = Scott's factor**2 times
    the per-dimension variance) so wildly different parameter scales are handled without a full
    covariance solve. ``bandwidth>0`` overrides Scott's factor; ``max_points`` subsamples the
    training draws for a tractable, low-memory ``(batch, n_points)`` kernel evaluation.
    """
    from keras import ops

    param_order = list(param_order)
    data = np.load(samples_path)
    cols = []
    for key in param_order:
        arr = np.asarray(data[key], dtype="float64").reshape(-1)
        if key in log10_keys:
            arr = np.log10(arr)
        cols.append(arr)
    X = np.stack(cols, axis=1)  # (N, d) in the network's native (log10) space
    X = X[np.all(np.isfinite(X), axis=1)]
    n_pts, d = X.shape
    if max_points and n_pts > int(max_points):
        idx = np.random.default_rng(seed).choice(n_pts, size=int(max_points), replace=False)
        X = X[idx]
    m_pts = X.shape[0]

    factor = float(bandwidth) if bandwidth and float(bandwidth) > 0 else m_pts ** (-1.0 / (d + 4))
    var = X.var(axis=0)
    h = np.maximum((factor**2) * var, 1e-12)  # diagonal bandwidth variances, (d,)
    inv_h_np = 1.0 / h

    # Center the data by its per-dimension mean before forming the kernel. The expanded
    # Mahalanobis form ``q - 2*cross + r`` suffers catastrophic float32 cancellation when a
    # dimension's magnitude is large relative to its bandwidth (e.g. log10(rho) ~ 7 with
    # bandwidth ~0.16: the ``q`` terms reach ~1900 while the O(1) differences that matter are
    # swamped). Working in ``u = theta - mu`` / ``Xc = X - mu`` keeps every term O(std) so the
    # softmax exponent is precise on GPU as well as CPU; the resulting gradient is unchanged
    # (a pure shift of both operands leaves ``theta - weighted_x`` invariant).
    mu_np = X.mean(axis=0)
    Xc = X - mu_np  # (M, d) centered
    Xc_t = ops.convert_to_tensor(Xc.astype("float32"))  # (M, d)
    Xc_T = ops.transpose(Xc_t)  # (d, M)
    mu = ops.convert_to_tensor(mu_np.astype("float32"))  # (d,)
    inv_h = ops.convert_to_tensor(inv_h_np.astype("float32"))  # (d,)
    r_pts = ops.convert_to_tensor(((Xc**2) * inv_h_np).sum(axis=1).astype("float32"))  # (M,)

    def score(x: Dict[str, np.ndarray], time=None) -> Dict[str, np.ndarray]:
        theta = ops.concatenate(
            [ops.reshape(ops.cast(x[k], "float32"), (-1, 1)) for k in param_order], axis=1
        )  # (B, d)
        u = theta - mu  # (B, d) centered query
        u_scaled = u * inv_h  # (B, d)
        q = ops.sum(u * u_scaled, axis=1, keepdims=True)  # (B, 1) = sum u^2 / h
        cross = ops.matmul(u_scaled, Xc_T)  # (B, M) = sum u * xc / h
        maha = -0.5 * (q - 2.0 * cross + ops.reshape(r_pts, (1, -1)))  # (B, M) log-kernel exponent
        w = ops.softmax(maha, axis=1)  # (B, M) responsibilities
        weighted_xc = ops.matmul(w, Xc_t)  # (B, d) = sum_j w_j (x_j - mu)
        grad = -(u - weighted_xc) * inv_h  # (B, d) = -H^{-1}(theta - sum_j w_j x_j)

        out = {}
        for i, key in enumerate(param_order):
            g = ops.reshape(grad[:, i], ops.shape(x[key]))
            out[key] = g if time is None else (1.0 - time) * g
        return out

    return score


def prior_score_from_kde_jax(
    samples_path: str,
    param_order,
    log10_keys: Container[str] = (),
    max_points: int = 4096,
    bandwidth: float = 0.0,
    seed: int = 0,
) -> Callable[[dict], dict]:
    """KDE compositional prior score via ``jax.scipy.stats.gaussian_kde`` + ``jax.grad``.

    An alternative to the hand-rolled :func:`prior_score_from_kde`. Same *space contract*: the
    KDE is fit on the training draws transformed to the network's native space (``log10`` on
    ``log10_keys``), so ``grad log p_KDE`` is directly the score bayesflow expects; the callable
    names ``time`` and applies the ``(1 - t)`` decay itself. The ONLY substantive differences
    from the custom version are (a) SciPy/JAX ``gaussian_kde`` uses a **full covariance**
    bandwidth (Scott's rule on the whole data covariance, so it whitens correlated parameters)
    where the custom version is **diagonal**, and (b) the gradient comes from autodiff
    (``jax.grad`` through ``logpdf``) rather than the closed-form softmax expression. On
    uncorrelated parameters the two coincide; on correlated ones they differ by the off-diagonal
    covariance terms.

    ``jax`` is imported lazily inside this function so the dependency is only ever incurred when
    this implementation is actually selected (``eval.prior_kde_impl=jax``). This path is only
    valid under the JAX keras backend, which HydraBFlow pins.

    ``bandwidth>0`` is passed to ``gaussian_kde`` as a scalar ``bw_method`` (== Scott factor
    override); ``0`` uses Scott's rule. ``max_points`` subsamples the training draws.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.stats import gaussian_kde

    param_order = list(param_order)
    data = np.load(samples_path)
    cols = []
    for key in param_order:
        arr = np.asarray(data[key], dtype="float64").reshape(-1)
        if key in log10_keys:
            arr = np.log10(arr)
        cols.append(arr)
    X = np.stack(cols, axis=1)  # (N, d) in the network's native (log10) space
    X = X[np.all(np.isfinite(X), axis=1)]
    n_pts, d = X.shape
    if max_points and n_pts > int(max_points):
        idx = np.random.default_rng(seed).choice(n_pts, size=int(max_points), replace=False)
        X = X[idx]

    dataset = jnp.asarray(X.T.astype("float32"))  # gaussian_kde wants (d, N)
    bw_method = float(bandwidth) if bandwidth and float(bandwidth) > 0 else "scott"
    kde = gaussian_kde(dataset, bw_method=bw_method)

    # logpdf of point j depends only on evaluation column j, so d/d(pts) of the summed logpdf
    # yields each column's own gradient -> a single grad call gives the (d, B) per-point score.
    grad_fn = jax.grad(lambda pts: jnp.sum(kde.logpdf(pts)))

    def score(x: Dict[str, np.ndarray], time=None) -> Dict[str, np.ndarray]:
        pts = jnp.stack(
            [jnp.reshape(jnp.asarray(x[k], dtype=jnp.float32), (-1,)) for k in param_order],
            axis=0,
        )  # (d, B)
        g = grad_fn(pts)  # (d, B)
        out = {}
        for i, key in enumerate(param_order):
            gi = jnp.reshape(g[i], jnp.shape(x[key]))
            out[key] = gi if time is None else (1.0 - time) * gi
        return out

    return score


def build_prior_score(cfg, simulator, log10_keys, param_order, seed: int = 0):
    """Select the compositional prior score from ``cfg.eval.prior_score`` (``spec``|``kde``).

    When ``prior_score=kde``, ``cfg.eval.prior_kde_impl`` chooses between the hand-rolled
    diagonal-bandwidth estimator (``diagonal``, default) and the ``jax.scipy.stats.gaussian_kde``
    full-covariance estimator (``jax``).
    """
    mode = str(getattr(getattr(cfg, "eval", None), "prior_score", "spec") or "spec")
    if mode == "kde":
        samples = str(getattr(cfg.eval, "prior_kde_samples", "") or "")
        if not samples:
            raise ValueError("eval.prior_score=kde requires eval.prior_kde_samples to be set")
        impl = str(getattr(cfg.eval, "prior_kde_impl", "diagonal") or "diagonal")
        builder = prior_score_from_kde_jax if impl == "jax" else prior_score_from_kde
        return builder(
            samples,
            param_order=param_order,
            log10_keys=log10_keys,
            max_points=int(getattr(cfg.eval, "prior_kde_max_points", 4096)),
            bandwidth=float(getattr(cfg.eval, "prior_kde_bandwidth", 0.0)),
            seed=seed,
        )
    return prior_score_from_spec(simulator.prior_spec_global, log10_keys=log10_keys)


def log10_keys_from_pipeline(pipeline) -> list:
    """Keys the ``log10_transform`` preprocessing step reparametrized, if present (else empty)."""
    step = pipeline.get_step("log10_transform") if pipeline is not None else None
    return list(step.keys) if step is not None else []


def flatten_members(data: Mapping[str, np.ndarray], m: int) -> Dict[str, np.ndarray]:
    """Reshape per-member arrays ``(n, m, ...) -> (n*m, ...)``; repeat group-level arrays
    (globals, rotation curve) ``m`` times so every flat row is complete.

    An array is treated as per-member when its second axis has length ``m`` and it has at least
    three dimensions — group-level observables must therefore not have ``m`` bins on axis 1.
    """
    out = {}
    for key, arr in data.items():
        arr = np.asarray(arr)
        if arr.ndim >= 3 and arr.shape[1] == m:
            out[key] = arr.reshape(arr.shape[0] * m, *arr.shape[2:])
        else:
            out[key] = np.repeat(arr, m, axis=0)
    return out


def group_members(data: Mapping[str, np.ndarray], n: int, m: int) -> Dict[str, np.ndarray]:
    """Inverse of :func:`flatten_members` for arrays of ``n*m`` rows -> ``(n, m, ...)``."""
    return {
        key: np.asarray(arr).reshape(n, m, *np.asarray(arr).shape[1:])
        for key, arr in data.items()
    }


def condition_keys(cfg) -> list:
    """Raw batch keys that act as sampling conditions (everything the adapter consumes except
    the inference targets)."""
    from hydrabflow.pipeline.adapter import adapter_keys

    targets = set(cfg.adapter.inference_variables)
    return [k for k in adapter_keys(cfg) if k not in targets]


def apply_augmentations_once(flat: Dict[str, np.ndarray], cfg, pipeline, seed: int):
    """Replay the configured augmentation chain once (fixed draw) on flattened rows."""
    from hydrabflow.augmentation.registry import build_augmentations

    augmentations = build_augmentations(
        cfg.augmentation, np.random.default_rng(seed), context={"pipeline": pipeline}
    )
    for aug in augmentations:
        flat = aug(flat)
    return {k: np.asarray(v) for k, v in flat.items()}
