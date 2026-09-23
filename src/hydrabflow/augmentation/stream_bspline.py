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
    # degree 0: indicator of [t_i, t_{i+1}), with the final interval closed so x == hi is inside
    b = ((x >= t[..., :-1]) & ((x < t[..., 1:]) | ((x == last) & (t[..., 1:] == last) & (t[..., :-1] < last)))).astype(x.dtype)
    for d in range(1, K + 1):
        den1 = t[..., d:-1] - t[..., :-d - 1]
        den2 = t[..., d + 1:] - t[..., 1:-d]
        a1 = jnp.where(den1 > 0, (x - t[..., :-d - 1]) / jnp.where(den1 > 0, den1, 1.0), 0.0)
        a2 = jnp.where(den2 > 0, (t[..., d + 1:] - x) / jnp.where(den2 > 0, den2, 1.0), 0.0)
        b = a1 * b[..., :-1] + a2 * b[..., 1:]
    return b


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
    t_all = jnp.asarray(clamped_knots(frames.track_edges))            # (S, n_int+2+2K)
    tv_all = jnp.asarray(clamped_knots(frames.vlos_edges))            # (S, n_int_v+2+2K)
    edges_all = jnp.asarray(frames.track_edges)                        # (S, n_int+2)
    vedges_all = jnp.asarray(frames.vlos_edges)
    u = jnp.linspace(0.0, 1.0, n_grid)

    def _fit_eval(B, w, y, Bg):
        """B (n,P,nb), w (n,P), y (n,P), Bg (n,G,nb) -> spline values on the grid (n,G)."""
        Bw = B * w[..., None]
        A = jnp.einsum("npi,npj->nij", Bw, B) + ridge * jnp.eye(B.shape[-1])
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
        lo, hi = edges[:, :1], edges[:, -1:]
        grid = lo + (hi - lo) * u[None]                                 # (n, G) uniform in the REAL range
        vgrid = vedges[:, :1] + (vedges[:, -1:] - vedges[:, :1]) * u[None]
        finite = jnp.all(jnp.isfinite(sim), -1)
        w = (attn.astype(bool) & finite & (phi1 >= lo) & (phi1 <= hi)).astype(jnp.float32)   # cut at the real range
        wv = w * vmask.astype(jnp.float32) * ((phi1 >= vedges[:, :1]) & (phi1 <= vedges[:, -1:]))
        phi1c = jnp.nan_to_num(phi1, nan=0.0)
        B, Bg = _basis(jnp, phi1c, t), _basis(jnp, grid, t)
        Bv, Bgv = _basis(jnp, phi1c, tv), _basis(jnp, vgrid, tv)
        tracks = [_fit_eval(B, w, jnp.nan_to_num(y), Bg) for y in (phi2, sim[..., par_c], mu_phi1, mu_phi2)]
        vlos = _fit_eval(Bv, wv, jnp.nan_to_num(sim[..., vlos_c]), Bgv)
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
