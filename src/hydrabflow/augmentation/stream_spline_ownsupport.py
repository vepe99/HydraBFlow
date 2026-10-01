"""Per-stream tracks fitted on each realization's OWN support, evaluated on the REAL φ1 grid (``stream_spline_ownsupport``).

The training twin of ``scripts/ppc_stream_splines.py`` (its defaults). Per row, in the published stream frame:

* core = the central ``spline_core_frac`` (0.85) φ1 quantile range of THAT row's attended stars (real or sim,
  after the observed-count subsample) — nothing is cut at the real φ1 range, only the sparse edges are dropped;
* φ2, μ_φ1, μ_φ2: penalized cubic spline over the core (uniformly extended knots, ``spline_knots`` interior,
  penalty ``λ Σ(Δ²c)²/h³`` ≈ ``λ ∫f''²``) with ``λ = spline_s_track · N_real · L_real³`` per stream (star count and
  φ1 length of the real members — the PPC's scipy smoothing-spline λ, fixed for every row);
* parallax: degree-``spline_parallax_deg`` (1) polynomial over the core stars;
* v_los: degree-``spline_vlos_deg`` (2) polynomial over the measured stars inside the core.

Everything is evaluated on ONE fixed grid per stream: ``bspline_grid_points`` (20) equal-count picks of the real
members' φ1 inside the real members' own core. A grid point outside the row's core (tracks, parallax) or outside
the [min, max] φ1 of its core measured stars (v_los) is invalid: value 0, flag -1. Layout == ``stream_bspline_grid``:

    batch["sim_summary"]  (n, G, 9) = [phi2, parallax, mu_phi1, mu_phi2, vlos, valid_track, valid_vlos, j, phi1]
"""
from __future__ import annotations

import numpy as np

from hydrabflow.registry import register_augmentation
from hydrabflow.augmentation.streams import _jax, _sim_key, _stream_ids_jax
from hydrabflow.augmentation.stream_summary import _DEFAULT_CHANNELS, _stream_frames
from hydrabflow.augmentation.stream_bspline import K, _basis, _real_member_xy, frame_observables


def _quantile(jnp, x, w, q):
    """Weighted-by-mask linear quantile per row, == np.quantile over the w>0 entries. x, w (n,P) -> (n,)."""
    xs = jnp.sort(jnp.where(w > 0, x, jnp.inf), -1)
    n = jnp.sum(w > 0, -1)
    pos = q * jnp.maximum(n - 1, 0)
    lo = jnp.floor(pos).astype(jnp.int32)
    hi = jnp.minimum(lo + 1, jnp.maximum(n - 1, 0))
    a = jnp.take_along_axis(xs, lo[:, None], -1)[:, 0]
    b = jnp.take_along_axis(xs, hi[:, None], -1)[:, 0]
    return jnp.where(n > 0, a + (pos - lo) * (b - a), 0.0)


def _solve(jnp, B, w, y, Bq, pen, ridge):
    """Weighted penalized LSQ per row: B (n,P,m), w/y (n,P), Bq (n,G,m), pen (n,m,m) -> values at the G points."""
    Bw = B * w[..., None]
    A = jnp.einsum("npi,npj->nij", Bw, B) + pen + ridge * jnp.eye(B.shape[-1])
    c = jnp.linalg.solve(A, jnp.einsum("npi,np->ni", Bw, y)[..., None])[..., 0]
    return jnp.einsum("ngi,ni->ng", Bq, c)


def _vander(jnp, x, c0, c1, deg):
    """Polynomial basis in u = (x - mid) / half over the row's range [c0, c1]: x (n,P) -> (n,P,deg+1)."""
    mid, half = 0.5 * (c0 + c1), jnp.maximum(0.5 * (c1 - c0), 1e-6)
    u = (x - mid[:, None]) / half[:, None]
    return jnp.stack([u ** d for d in range(deg + 1)], -1)


@register_augmentation("stream_spline_ownsupport")
def _stream_spline_ownsupport(params, rng, context=None):
    jax, jnp = _jax()
    obs_key = _sim_key(params)
    summary_key = str(params.get("summary_key", "sim_summary"))
    n_grid = int(params.get("bspline_grid_points", 20))
    core_frac = float(params.get("spline_core_frac", 0.85))
    n_int = int(params.get("spline_knots", 20))
    s_track = float(params.get("spline_s_track", 1e-4))
    plx_deg = int(params.get("spline_parallax_deg", 1))
    vlos_deg = int(params.get("spline_vlos_deg", 2))
    ridge = float(params.get("bspline_ridge", 1e-3))
    min_track = int(params.get("spline_min_stars", 8))       # below this many core stars the whole row is invalid
    channels = dict(_DEFAULT_CHANNELS)
    channels.update({k: int(v) for k, v in (params.get("summary_channels", {}) or {}).items() if k in channels})
    par_c, vlos_c = channels["parallax"], channels["vlos"]
    qq = ((1 - core_frac) / 2, (1 + core_frac) / 2) if 0 < core_frac < 1 else (0.0, 1.0)

    # real members: the fixed per-stream φ1 grid (equal-count picks inside the real core) and the track λ
    frames = _stream_frames(params, 1, 1, channels)
    S = frames.R.shape[0]
    grid_np, lam_np = np.zeros((S, n_grid)), np.ones(S)
    for j, obs in enumerate(_real_member_xy(params, frames, channels)):
        if obs is None:
            continue
        x = np.sort(obs[0][0])
        c0, c1 = np.quantile(x, qq)
        xc = x[(x >= c0) & (x <= c1)]
        grid_np[j] = xc[np.round(np.linspace(0, len(xc) - 1, n_grid)).astype(int)]
        lam_np[j] = s_track * len(x) * np.ptp(x) ** 3
    R_all, grid_all, lam_all = jnp.asarray(frames.R), jnp.asarray(grid_np), jnp.asarray(lam_np)
    D = np.diff(np.eye(n_int + K + 1), 2, axis=0)
    DtD = jnp.asarray(D.T @ D)
    offs = jnp.arange(-K, n_int + K + 2, dtype=jnp.float32)

    @jax.jit
    def _run(sim, attn, vmask, j):
        phi1, phi2, mu_phi1, mu_phi2 = frame_observables(jnp, R_all[j], sim, channels)
        finite = jnp.all(jnp.isfinite(sim), -1) & jnp.isfinite(phi1)
        w = (attn.astype(bool) & finite).astype(jnp.float32)
        phi1 = jnp.nan_to_num(phi1)
        c0, c1 = _quantile(jnp, phi1, w, qq[0]), _quantile(jnp, phi1, w, qq[1])
        wc = w * ((phi1 >= c0[:, None]) & (phi1 <= c1[:, None]))
        wv = wc * vmask.astype(jnp.float32)
        grid = grid_all[j]                                                    # (n, G), the real φ1 grid

        # φ2, μ_φ1, μ_φ2: P-spline on uniformly extended knots over the row's core
        h = jnp.maximum((c1 - c0) / (n_int + 1), 1e-6)
        t = c0[:, None] + h[:, None] * offs[None]
        B, Bq = _basis(jnp, phi1, t), _basis(jnp, jnp.clip(grid, c0[:, None], c1[:, None]), t)
        pen = (lam_all[j] / h ** 3)[:, None, None] * DtD[None]
        spl = [_solve(jnp, B, wc, jnp.nan_to_num(y), Bq, pen, ridge) for y in (phi2, mu_phi1, mu_phi2)]
        # parallax and v_los: low-order polynomials (v_los over the measured core stars and their own range)
        V, Vq = _vander(jnp, phi1, c0, c1, plx_deg), _vander(jnp, grid, c0, c1, plx_deg)
        plx = _solve(jnp, V, wc, jnp.nan_to_num(sim[..., par_c]), Vq, 0.0, 1e-6)
        v0 = jnp.min(jnp.where(wv > 0, phi1, jnp.inf), -1)
        v1 = jnp.max(jnp.where(wv > 0, phi1, -jnp.inf), -1)
        U, Uq = _vander(jnp, phi1, v0, v1, vlos_deg), _vander(jnp, grid, v0, v1, vlos_deg)
        vlos = _solve(jnp, U, wv, jnp.nan_to_num(sim[..., vlos_c]), Uq, 0.0, 1e-6)

        in_core = (grid >= c0[:, None]) & (grid <= c1[:, None]) & (jnp.sum(wc, -1) >= min_track)[:, None]
        in_v = (grid >= v0[:, None]) & (grid <= v1[:, None]) & (jnp.sum(wv, -1) >= vlos_deg + 2)[:, None]
        tracks = [jnp.where(in_core, y, 0.0) for y in (spl[0], plx, spl[1], spl[2])]
        vlos = jnp.where(in_v, vlos, 0.0)
        j_bc = jnp.broadcast_to(j.reshape(-1, 1).astype(jnp.float32), grid.shape)
        out = jnp.stack(tracks + [vlos, jnp.where(in_core, 1.0, -1.0), jnp.where(in_v, 1.0, -1.0), j_bc, grid], -1)
        return jnp.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)

    def aug(batch):
        sim = jnp.asarray(batch[obs_key])
        attn = jnp.asarray(batch["attention_mask"])[:, 0, :]
        vmask = jnp.asarray(batch["vlos_mask"])[:, 0, :]
        batch[summary_key] = _run(sim, attn, vmask, _stream_ids_jax(batch))
        return batch

    return aug
