"""Per-stream **fixed-knot cubic LSQ B-spline tracks** as a φ1 time series (``stream_bspline_grid``).

The alternative to the binned medians/dispersions of :mod:`stream_summary`: each observable
(φ2, parallax, μ_φ1, μ_φ2 over the attended stars; v_los over the MEASURED stars) is fitted vs φ1
with a cubic B-spline on a knot vector fixed per stream by the REAL members' φ1 quantiles, then
evaluated on ``bspline_grid_points`` uniform φ1 points inside the real φ1 range. The spline is cut
at the edge of the observational range: stars outside it get zero weight and nothing is evaluated
beyond it. Every realization therefore yields the same feature layout, on GPU, in JAX:

    batch["sim_summary"]  (n, G, 9) = [phi2, parallax, mu_phi1, mu_phi2, vlos,
                                        valid_track, valid_vlos, j, phi1]      (phi1 last, time_axis=-1)

The fit is the weighted normal-equation solve ``(BᵀWB + λI) c = BᵀWy`` per row and observable
(basis count ~ interior knots + 4, so a tiny batched solve); ``λ`` (``bspline_ridge``) keeps empty
knot spans finite, and ``valid_*`` = ±1 flags whether the knot span containing each grid point holds
at least ``summary_min_count`` stars — values in invalid spans are zeroed, as in the grid estimator.
The frame and knots come from :class:`stream_summary.StreamFrames` (``k_track = interior+1`` quantile
edges == the knot vector), so sim and real share one estimator.

``bspline_fit: smoothing`` swaps the fixed-knot fit for a **penalized (P-)spline**: a dense uniform
knot vector (``bspline_smoothing_knots`` interior knots over the real φ1 range, ``bspline_smoothing_vlos_knots``
for v_los) with the second-difference penalty ``λ Σ(Δ²c)²/h³`` (≈ ``λ ∫f''²``, the smoothing-spline
penalty, ``h`` = knot spacing). ``λ`` is per (stream, observable): ``bspline_smoothing_lambda: gcv``
(default) takes scipy's GCV choice for the smoothing spline of the real members (unit weights, exactly
`scripts/ppc_bspline_nn.py --fit smoothing`), a float pins it. The ``valid_*``
flags still use the quantile edges, so the layout is unchanged and the LSQ model configs apply.
"""
from __future__ import annotations

import numpy as np

from hydrabflow.registry import register_augmentation
from hydrabflow.augmentation.streams import _jax, _sim_key, _stream_ids_jax
from hydrabflow.augmentation.stream_summary import _DEFAULT_CHANNELS, _stream_frames

K = 3


def clamped_knots(edges: np.ndarray) -> np.ndarray:
    """(S, m) quantile edges (ends = real φ1 range) -> (S, m + 2K) clamped cubic knot vector."""
    lo, hi = edges[:, :1], edges[:, -1:]
    return np.concatenate([np.repeat(lo, K, 1), edges, np.repeat(hi, K, 1)], axis=1)


def _basis(jnp, x, t):
    """Cox-de Boor cubic B-spline basis. ``x`` (..., P), ``t`` (..., m) broadcastable -> (..., P, m-K-1)."""
    x = x[..., :, None]                      # (..., P, 1)
    t = t[..., None, :]                      # (..., 1, m)
    last = t[..., -1:]
    # a grid end computed as lo + (hi - lo) * 1.0 can land one ulp above the last knot, where every basis
    # function is zero; clip it back (points beyond the range carry zero weight anyway)
    x = jnp.minimum(x, last)
    # degree 0: indicator of [t_i, t_{i+1}), with the final interval closed so x == hi is inside
    b = ((x >= t[..., :-1]) & ((x < t[..., 1:]) | ((x == last) & (t[..., 1:] == last) & (t[..., :-1] < last)))).astype(x.dtype)
    for d in range(1, K + 1):
        den1 = t[..., d:-1] - t[..., :-d - 1]
        den2 = t[..., d + 1:] - t[..., 1:-d]
        a1 = jnp.where(den1 > 0, (x - t[..., :-d - 1]) / jnp.where(den1 > 0, den1, 1.0), 0.0)
        a2 = jnp.where(den2 > 0, (t[..., d + 1:] - x) / jnp.where(den2 > 0, den2, 1.0), 0.0)
        b = a1 * b[..., :-1] + a2 * b[..., 1:]
    return b


def _uniform_knots(edges: np.ndarray, n_int: int) -> np.ndarray:
    """(S, m) edges -> (S, n_int+2+2K) clamped knots, uniform between the real φ1 ends."""
    lo, hi = edges[:, :1], edges[:, -1:]
    return clamped_knots(lo + (hi - lo) * np.linspace(0.0, 1.0, n_int + 2)[None])


def _second_diff_penalty(t: np.ndarray) -> np.ndarray:
    """(S, m) knot vector -> (S, nb, nb) ``D₂ᵀD₂ / h³`` for its ``nb = m-K-1`` cubic basis functions."""
    nb = t.shape[1] - K - 1
    D = np.diff(np.eye(nb), 2, axis=0)
    h = (t[:, -1] - t[:, 0]) / (nb - K)                       # uniform interior spacing
    return (D.T @ D)[None] / h[:, None, None] ** 3


def _np_basis(x, t):
    import jax.numpy as jnp
    return np.asarray(_basis(jnp, jnp.asarray(x), jnp.asarray(t)))


def _gcv_lambda(x, y, t, fallback=100.0):
    """λ chosen by scipy's GCV for the cubic smoothing spline of ``y(x)`` (unit weights, near-duplicate
    ``x`` merged as in `scripts/ppc_bspline_nn.py`). The P-spline penalty approximates ``λ ∫f''²`` on the
    same scale, so the value transfers (checked: rms difference to scipy's curve 0.3-15 % of its scale)."""
    import scipy.interpolate._bsplines as _bs
    from scipy.interpolate import make_smoothing_spline

    m = np.isfinite(x) & np.isfinite(y) & (x >= t[0]) & (x <= t[-1])
    x, y = x[m], y[m]
    if len(x) < 6:
        return float(fallback)
    xu, inv = np.unique(np.round(x, 4), return_inverse=True)
    cnt = np.bincount(inv)
    yu = np.bincount(inv, y) / cnt
    got, orig = [], _bs._compute_optimal_gcv_parameter

    def capture(*a, **k):
        got.append(orig(*a, **k))
        return got[-1]

    _bs._compute_optimal_gcv_parameter = capture
    try:
        make_smoothing_spline(xu, yu, w=cnt.astype(float))
    except (ValueError, np.linalg.LinAlgError):
        return float(fallback)
    finally:
        _bs._compute_optimal_gcv_parameter = orig
    return float(got[-1]) if got else float(fallback)


def _np_project(R, s, ch):
    """Real members (P, 6) -> phi1, phi2, parallax, mu_phi1, mu_phi2, vlos (numpy twin of ``_run``)."""
    ra, dec = np.radians(s[:, ch["ra"]]), np.radians(s[:, ch["dec"]])
    n = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1) @ R.T
    phi1 = np.arctan2(n[:, 1], n[:, 0])
    phi2 = np.arcsin(np.clip(n[:, 2], -1, 1))
    e = np.stack([-np.sin(ra), np.cos(ra), 0 * ra], -1)
    m = np.stack([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], -1)
    v = (s[:, ch["mu_ra"], None] * e + s[:, ch["mu_dec"], None] * m) @ R.T
    ep = np.stack([-np.sin(phi1), np.cos(phi1), 0 * phi1], -1)
    mp = np.stack([-np.sin(phi2) * np.cos(phi1), -np.sin(phi2) * np.sin(phi1), np.cos(phi2)], -1)
    return (np.degrees(phi1), np.degrees(phi2), s[:, ch["parallax"]], np.sum(v * ep, -1), np.sum(v * mp, -1), s[:, ch["vlos"]])


def _real_gcv_lambdas(params, frames, t_all, tv_all, channels):
    """(S, 5) GCV λ per stream × [phi2, parallax, mu_phi1, mu_phi2, vlos] from the real members."""
    import os
    real_file = params.get("real_streams_file", os.path.join(params.get("resources_dir", "assets/gaia"),
                                                             "gaia_observed_streams_6Dwitherrors_cutNGC3201.npz"))
    d = np.load(real_file)
    sim = np.asarray(d["sim_data_projected"], dtype=float)
    sim = sim[0] if sim.ndim == 4 else sim
    am = np.asarray(d["attention_mask"])
    am = am[:, 0, :] if am.ndim == 3 else am
    vm = np.asarray(d["vlos_mask"])
    vm = vm[:, 0, :] if vm.ndim == 3 else vm
    jarr = np.asarray(d["j"]).reshape(-1).astype(int)
    S = frames.R.shape[0]
    lam = np.ones((S, 5))
    for row in range(sim.shape[0]):
        j = int(jarr[row])
        if j >= S:
            continue
        mem = am[row].astype(bool)
        vmeas = vm[row].astype(bool) & mem
        phi1, *obs = _np_project(frames.R[j], sim[row][mem], channels)
        for k, y in enumerate(obs[:4]):
            lam[j, k] = _gcv_lambda(phi1, y, t_all[j])
        p1v, *_, vlos = _np_project(frames.R[j], sim[row][vmeas], channels)
        lam[j, 4] = _gcv_lambda(p1v, vlos, tv_all[j])
    return lam


@register_augmentation("stream_bspline_grid")
def _stream_bspline_grid(params, rng, context=None):
    jax, jnp = _jax()
    obs_key = _sim_key(params)
    summary_key = str(params.get("summary_key", "sim_summary"))
    n_int = int(params.get("bspline_interior_knots", 5))
    n_int_v = int(params.get("bspline_vlos_interior_knots", 1))
    n_grid = int(params.get("bspline_grid_points", 20))
    ridge = float(params.get("bspline_ridge", 1e-3))
    min_count = int(params.get("summary_min_count", 3))
    channels = dict(_DEFAULT_CHANNELS)
    channels.update({k: int(v) for k, v in (params.get("summary_channels", {}) or {}).items() if k in channels})
    ra_c, dec_c = channels["ra"], channels["dec"]
    par_c, mura_c, mudec_c, vlos_c = channels["parallax"], channels["mu_ra"], channels["mu_dec"], channels["vlos"]

    # k_track = interior + 1 quantile edges of the real members -> the knot vector, ends = real φ1 range
    frames = _stream_frames(params, n_int + 1, n_int_v + 1, channels)
    R_all = jnp.asarray(frames.R)
    fit = str(params.get("bspline_fit", "lsq")).lower()
    if fit == "lsq":
        t_np, tv_np = clamped_knots(frames.track_edges), clamped_knots(frames.vlos_edges)
        pen_np = np.zeros((frames.R.shape[0], 5, t_np.shape[1] - K - 1, t_np.shape[1] - K - 1))
        penv_np = np.zeros((frames.R.shape[0], 5, tv_np.shape[1] - K - 1, tv_np.shape[1] - K - 1))
    elif fit == "smoothing":
        t_np = _uniform_knots(frames.track_edges, int(params.get("bspline_smoothing_knots", 20)))
        tv_np = _uniform_knots(frames.vlos_edges, int(params.get("bspline_smoothing_vlos_knots", 6)))
        om, omv = _second_diff_penalty(t_np), _second_diff_penalty(tv_np)
        lam_cfg = params.get("bspline_smoothing_lambda", "gcv")
        if str(lam_cfg).lower() == "gcv":
            lam = _real_gcv_lambdas(params, frames, t_np, tv_np, channels)
        else:
            lam = np.full((frames.R.shape[0], 5), float(lam_cfg))
        pen_np = lam[:, :4, None, None] * om[:, None]                # (S, 4, nb, nb)
        penv_np = lam[:, 4:, None, None] * omv[:, None]              # (S, 1, nbv, nbv)
    else:
        raise ValueError(f"bspline_fit must be 'lsq' or 'smoothing', got {fit!r}")
    t_all, tv_all = jnp.asarray(t_np), jnp.asarray(tv_np)            # (S, m), (S, mv)
    pen_all, penv_all = jnp.asarray(pen_np), jnp.asarray(penv_np)
    edges_all = jnp.asarray(frames.track_edges)                        # (S, n_int+2)
    vedges_all = jnp.asarray(frames.vlos_edges)
    u = jnp.linspace(0.0, 1.0, n_grid)

    def _fit_eval(B, w, y, Bg, pen):
        """B (n,P,nb), w (n,P), y (n,P), Bg (n,G,nb), pen (n,nb,nb) -> spline values on the grid (n,G)."""
        Bw = B * w[..., None]
        A = jnp.einsum("npi,npj->nij", Bw, B) + ridge * jnp.eye(B.shape[-1]) + pen
        rhs = jnp.einsum("npi,np->ni", Bw, y)
        c = jnp.linalg.solve(A, rhs[..., None])[..., 0]
        return jnp.einsum("ngi,ni->ng", Bg, c)

    def _span_valid(edges, phi1, w, grid):
        """±1 per grid point: does the knot span containing it hold >= min_count weighted stars?"""
        span_star = jnp.clip(jnp.sum(phi1[..., None] >= edges[:, None, :], -1) - 1, 0, edges.shape[1] - 2)
        span_grid = jnp.clip(jnp.sum(grid[..., None] >= edges[:, None, :], -1) - 1, 0, edges.shape[1] - 2)
        counts = jnp.sum((span_star[:, :, None] == jnp.arange(edges.shape[1] - 1)[None, None, :]) * w[..., None], 1)  # (n, spans)
        return jnp.where(jnp.take_along_axis(counts, span_grid, 1) >= min_count, 1.0, -1.0)

    @jax.jit
    def _run(sim, attn, vmask, j):
        ra, dec = sim[..., ra_c], sim[..., dec_c]
        rar, decr = jnp.radians(ra), jnp.radians(dec)
        n_vec = jnp.stack([jnp.cos(decr) * jnp.cos(rar), jnp.cos(decr) * jnp.sin(rar), jnp.sin(decr)], -1)
        Rj = R_all[j]
        npr = jnp.einsum("nij,npj->npi", Rj, n_vec)
        phi1 = jnp.degrees(jnp.arctan2(npr[..., 1], npr[..., 0]))
        phi2 = jnp.degrees(jnp.arcsin(jnp.clip(npr[..., 2], -1.0, 1.0)))
        e = jnp.stack([-jnp.sin(rar), jnp.cos(rar), jnp.zeros_like(rar)], -1)
        m = jnp.stack([-jnp.sin(decr) * jnp.cos(rar), -jnp.sin(decr) * jnp.sin(rar), jnp.cos(decr)], -1)
        v = sim[..., mura_c][..., None] * e + sim[..., mudec_c][..., None] * m
        vpr = jnp.einsum("nij,npj->npi", Rj, v)
        p1, p2 = jnp.radians(phi1), jnp.radians(phi2)
        ep = jnp.stack([-jnp.sin(p1), jnp.cos(p1), jnp.zeros_like(p1)], -1)
        mp = jnp.stack([-jnp.sin(p2) * jnp.cos(p1), -jnp.sin(p2) * jnp.sin(p1), jnp.cos(p2)], -1)
        mu_phi1, mu_phi2 = jnp.sum(vpr * ep, -1), jnp.sum(vpr * mp, -1)

        t, tv, edges, vedges = t_all[j], tv_all[j], edges_all[j], vedges_all[j]
        pen, penv = pen_all[j], penv_all[j]                             # (n, 4, nb, nb), (n, 1, nbv, nbv)
        lo, hi = edges[:, :1], edges[:, -1:]
        grid = lo + (hi - lo) * u[None]                                 # (n, G) uniform in the REAL range
        vgrid = vedges[:, :1] + (vedges[:, -1:] - vedges[:, :1]) * u[None]
        finite = jnp.all(jnp.isfinite(sim), -1)
        w = (attn.astype(bool) & finite & (phi1 >= lo) & (phi1 <= hi)).astype(jnp.float32)   # cut at the real range
        wv = w * vmask.astype(jnp.float32) * ((phi1 >= vedges[:, :1]) & (phi1 <= vedges[:, -1:]))
        phi1c = jnp.nan_to_num(phi1, nan=0.0)
        B, Bg = _basis(jnp, phi1c, t), _basis(jnp, grid, t)
        Bv, Bgv = _basis(jnp, phi1c, tv), _basis(jnp, vgrid, tv)
        tracks = [_fit_eval(B, w, jnp.nan_to_num(y), Bg, pen[:, k])
                  for k, y in enumerate((phi2, sim[..., par_c], mu_phi1, mu_phi2))]
        vlos = _fit_eval(Bv, wv, jnp.nan_to_num(sim[..., vlos_c]), Bgv, penv[:, 0])
        valid_t = _span_valid(edges, phi1c, w, grid)
        valid_v = _span_valid(vedges, phi1c, wv, vgrid)
        tracks = [jnp.where(valid_t > 0, tr, 0.0) for tr in tracks]
        vlos = jnp.where(valid_v > 0, vlos, 0.0)
        j_bc = jnp.broadcast_to(j.reshape(-1, 1).astype(jnp.float32), grid.shape)
        out = jnp.stack(tracks + [vlos, valid_t, valid_v, j_bc, grid], -1)
        return jnp.nan_to_num(out, nan=0.0).astype(jnp.float32)

    def aug(batch):
        sim = jnp.asarray(batch[obs_key])
        attn = jnp.asarray(batch["attention_mask"])[:, 0, :]
        vmask = jnp.asarray(batch["vlos_mask"])[:, 0, :]
        batch[summary_key] = _run(sim, attn, vmask, _stream_ids_jax(batch))
        return batch

    return aug
