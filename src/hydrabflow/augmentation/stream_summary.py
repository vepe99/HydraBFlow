"""Hand-crafted per-stream summary statistics in a data-driven stream-aligned frame.

Motivation (see CLAUDE.md ``TODO — summary-statistics observables`` and ``summary_stats_extentions.md``):
the learned SetTransformer summaries of the raw star cloud extrapolate arbitrarily off-manifold on
real Gaia data. Injecting physically-motivated, low-dimensional, potential-sensitive summaries
(Bonaca & Hogg 2018's 6D stream tracks) gives the network a robust backbone.

This augmentation writes ``batch["sim_summary"]`` — a fixed-length per-stream vector of **binned
median + std of the stream-frame tracks**:

1. A great-circle frame ``R_j`` is fitted **once** per stream from its *real* Gaia members (the
   normal to the best-fit plane = smallest-eigenvalue eigenvector of ``Σ n nᵀ``). Positions rotate
   to ``(φ1, φ2)``; proper motions rotate exactly into ``(μ_φ1, μ_φ2)``. parallax / v_los are
   frame-invariant. Fitting from the real members (a fixed observational property) makes the sim
   and real summaries live in the *same* frame → directly comparable.
2. φ1 is binned with per-stream equal-count quantile edges of the real members. Position/pm tracks
   use ``summary_track_bins`` bins; v_los uses fewer (``summary_vlos_bins``) since it is measured
   for far fewer stars — and v_los stats use **measured stars only** (native missing-v_los
   handling, no imputation).
3. Per bin: median + dispersion of ``{φ2, parallax, μ_φ1, μ_φ2}`` (track bins) and of ``v_los``
   (vlos bins), the per-bin **occupancy** counts, plus a few scalars (measured-vlos fraction,
   attended fraction, φ1 extent, arm asymmetry, out-of-range fraction) and the stream index ``j``
   (so the summary MLP is stream-aware).

The dispersion estimator matters more than it looks. The φ1 bin edges are equal-count quantiles of
the *real* members, so real bins hold exactly ``N/K`` stars by construction while simulated ones do
not — a simulated stream that does not span the observational window ends up with sparse bins. With
the population (``ddof=0``) standard deviation and empty bins filled with 0, that occupancy
difference alone biases simulated dispersions low (0.72× at 3 stars, 0.57× at 2, exactly 0 at 1),
which reads as "the simulated streams are too cold". :func:`_binned_stats` therefore uses ``ddof=1``
(or a MAD scale), returns NaN below ``summary_min_count``, and exports occupancy explicitly; and
:func:`_bin_assignment` drops out-of-range stars instead of clipping them into the end bins. See
``summary_scale`` / ``summary_min_count`` / ``summary_include_occupancy``.

The frame + edges are baked once (NumPy) and the per-batch projection + binning runs in ``jax.jit``
(astropy/gala transforms are far too slow per batch). Runs **after** ``log10_vcirc`` and **before**
the feature concatenations, so the observable still has its 6 raw channels
``[ra, dec, parallax, μ_ra_cosdec, μ_dec, v_los]``.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from hydrabflow.registry import register_augmentation
from hydrabflow.augmentation.streams import _jax, _sim_key, _stream_ids_jax

_FRAME_CACHE: Dict[str, "StreamFrames"] = {}

# Default channel layout of the observable at summary time (before concatenations).
_DEFAULT_CHANNELS = {"ra": 0, "dec": 1, "parallax": 2, "mu_ra": 3, "mu_dec": 4, "vlos": 5}


# ------------------------------------------------------------------------------------------- #
# NumPy great-circle frame fit (setup-time, once per stream)
# ------------------------------------------------------------------------------------------- #


def _np_unit_vec(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def _np_fit_frame(ra, dec):
    """Rows of R are [x, y, z]: z = normal to best-fit plane (great-circle pole), x = mean member
    direction projected off z, y = z × x. Proper rotation (det = +1, right-handed)."""
    n = _np_unit_vec(ra, dec)
    _, v = np.linalg.eigh(n.T @ n)
    pole = v[:, 0]
    mean = n.mean(0)
    mean = mean / np.linalg.norm(mean)
    x = mean - (mean @ pole) * pole
    x = x / np.linalg.norm(x)
    y = np.cross(pole, x)
    return np.stack([x, y, pole], 0)


def _np_phi1(R, ra, dec):
    npr = _np_unit_vec(ra, dec) @ R.T
    return np.degrees(np.arctan2(npr[:, 1], npr[:, 0]))


class StreamFrames:
    """Per-stream great-circle rotation ``R_j`` and φ1 bin edges, fitted once from the real npz."""

    def __init__(self, params: dict, k_track: int, k_vlos: int, channels: dict) -> None:
        real_file = params.get(
            "real_streams_file",
            os.path.join(
                params.get("resources_dir", "assets/gaia"),
                "gaia_observed_streams_6Dwitherrors_cutNGC3201.npz",
            ),
        )
        target = {str(k): int(v) for k, v in params["target_streams"].items()}
        n_streams = max(target.values()) + 1
        # `summary_bin_edges`: "quantile" (default) = equal-COUNT edges of the real members, so the
        # real occupancy is flat by construction and out of distribution for every sim;
        # "uniform" = equal-WIDTH edges over the same phi1 span, so occupancy becomes a linear
        # density — a physical observable both sides can share.
        self.edge_mode = str(params.get("summary_bin_edges", "quantile")).lower()
        if self.edge_mode not in ("quantile", "uniform"):
            raise ValueError(f"summary_bin_edges must be 'quantile' or 'uniform', got {self.edge_mode!r}")

        d = np.load(real_file)
        sim = np.asarray(d["sim_data_projected"], dtype=float)
        sim = sim[0] if sim.ndim == 4 else sim  # (rows, P, 6)
        am = np.asarray(d["attention_mask"])
        am = am[:, 0, :] if am.ndim == 3 else am  # (rows, P)
        vm = np.asarray(d["vlos_mask"])
        vm = vm[:, 0, :] if vm.ndim == 3 else vm  # (rows, P)
        jarr = np.asarray(d["j"]).reshape(-1).astype(int)  # (rows,)

        # `summary_frame`: "fit" (default) = a great circle fitted to the real members; "streamfinder" = the
        # PUBLISHED Ibata+2024 Table 3 frame, which does not move with the member catalogue
        frame_mode = str(params.get("summary_frame", "fit")).lower()
        if frame_mode not in ("fit", "streamfinder"):
            raise ValueError(f"summary_frame must be 'fit' or 'streamfinder', got {frame_mode!r}")
        if frame_mode == "streamfinder":
            from hydrabflow.simulators.stream_frame import frames as _published
            pub = _published(names=tuple(target))
        name_of = {v: k for k, v in target.items()}
        ra_c, dec_c = channels["ra"], channels["dec"]
        self.R = np.repeat(np.eye(3)[None], n_streams, axis=0)
        self.track_edges = np.repeat(np.linspace(-1, 1, k_track + 1)[None], n_streams, axis=0)
        self.vlos_edges = np.repeat(np.linspace(-1, 1, k_vlos + 1)[None], n_streams, axis=0)

        for row in range(sim.shape[0]):
            j = int(jarr[row])
            if j >= n_streams:
                continue
            mem = am[row].astype(bool)
            if mem.sum() < max(k_track + 1, 3):
                continue
            s = sim[row][mem]
            vmeas = (vm[row].astype(bool) & mem)[mem]
            R_j = pub[name_of[j]] if frame_mode == "streamfinder" else _np_fit_frame(s[:, ra_c], s[:, dec_c])
            phi1 = _np_phi1(R_j, s[:, ra_c], s[:, dec_c])
            self.R[j] = R_j
            src = phi1[vmeas] if vmeas.sum() > k_vlos else phi1
            if self.edge_mode == "uniform":
                self.track_edges[j] = np.linspace(phi1.min(), phi1.max(), k_track + 1)
                self.vlos_edges[j] = np.linspace(src.min(), src.max(), k_vlos + 1)
            else:
                self.track_edges[j] = np.quantile(phi1, np.linspace(0, 1, k_track + 1))
                self.vlos_edges[j] = np.quantile(src, np.linspace(0, 1, k_vlos + 1))


def _stream_frames(params: dict, k_track: int, k_vlos: int, channels: dict) -> StreamFrames:
    key = json.dumps(
        {
            "real": params.get("real_streams_file"),
            "resources": params.get("resources_dir"),
            "target": {str(k): int(v) for k, v in params["target_streams"].items()},
            "kt": k_track,
            "kv": k_vlos,
            "ch": channels,
            "edges": str(params.get("summary_bin_edges", "quantile")),
            "frame": str(params.get("summary_frame", "fit")),
        },
        sort_keys=True,
    )
    if key not in _FRAME_CACHE:
        _FRAME_CACHE[key] = StreamFrames(params, k_track, k_vlos, channels)
    return _FRAME_CACHE[key]


# ------------------------------------------------------------------------------------------- #
# Shared per-bin estimator
# ------------------------------------------------------------------------------------------- #


def _bin_assignment(jax, jnp, edges, phi1, k):
    """Per-star bin index plus an ``in_range`` flag.

    ``searchsorted`` returns 0 for stars below the first edge and ``k+1`` for stars above the last,
    so ``idx = ... - 1`` is outside ``[0, k)`` exactly for out-of-range stars. We return the flag
    instead of clipping: clipping piles every star beyond the real φ1 range into the end bins,
    which inflates their dispersion while starving the interior ones.
    """
    idx = jax.vmap(lambda ed, x: jnp.searchsorted(ed, x, side="right"))(edges, phi1) - 1
    in_range = (idx >= 0) & (idx < k)
    return jnp.clip(idx, 0, k - 1), in_range


def _binned_stats(jnp, vals, in_mask, scale: str, min_count: int):
    """Per-bin ``(median, dispersion, count)`` over the masked members.

    ``vals`` is ``(n, P)``, ``in_mask`` is ``(n, K, P)`` bool; outputs are ``(n, K)``.

    Two deliberate choices, both fixing biases that made simulated streams look colder than they
    are (the estimator is the same for sim and real, but the *occupancy* is not — real bins are
    equal-count by construction, simulated ones are not):

    * the dispersion uses ``ddof=1`` (or a MAD scale). The population ``ddof=0`` form is biased low
      by ``sqrt((N-1)/N)`` — 0.72 at N=3, 0.57 at N=2, and exactly 0 at N=1;
    * bins with fewer than ``min_count`` members yield **NaN**, not 0. The caller records occupancy
      in its own channel and substitutes a finite sentinel, so an empty bin can no longer pass for
      an on-track, zero-dispersion one (for φ2 a fabricated 0 *is* the on-track value, which made
      this invisible to inspection).
    """
    m = jnp.where(in_mask, vals[:, None, :], jnp.nan)
    count = in_mask.sum(-1)
    med = jnp.nanmedian(m, axis=2)
    if scale == "mad":
        disp = 1.4826 * jnp.nanmedian(jnp.abs(m - med[..., None]), axis=2)
    else:
        disp = jnp.nanstd(m, axis=2, ddof=1)
    med = jnp.where(count >= 1, med, jnp.nan)
    disp = jnp.where(count >= max(int(min_count), 2), disp, jnp.nan)
    return med, disp, count.astype(jnp.float32)


def _estimator_params(params) -> tuple[str, int, bool]:
    """``(scale, min_count, include_occupancy)`` from the augmentation params."""
    scale = str(params.get("summary_scale", "std")).lower()
    if scale not in ("std", "mad"):
        raise ValueError(f"summary_scale must be 'std' or 'mad', got {scale!r}")
    return scale, int(params.get("summary_min_count", 3)), bool(
        params.get("summary_include_occupancy", True)
    )


def occupancy_encoding(params) -> str:
    """``summary_occupancy``: how the per-bin occupancy channels are written.

    * ``counts`` (default, the legacy layout): the raw member count per bin. Readable by the PPC
      scripts and usable as a *feature* by a plain backbone — but NOT by the masked backbones when
      ``training.standardize`` includes ``summary_variables``: BayesFlow standardizes the whole grid
      before the summary network, so a ``count >= min_count`` test inside the network runs on
      z-scores and declares almost every bin empty (measured 2026-09-22: 1.7 % of bins "valid"
      against a true 94 %; the v5 ``default``/``nodisp`` runs and every earlier masked-backbone run
      trained on an essentially zeroed stream grid).
    * ``valid``: ``+1`` where the bin holds at least ``summary_min_count`` members, ``-1`` otherwise.
      Standardization is affine with a positive scale, so the SIGN survives it and the masked
      backbones recover the exact validity mask with ``>= 0``.
    """
    enc = str(params.get("summary_occupancy", "counts")).lower()
    if enc not in ("counts", "valid"):
        raise ValueError(f"summary_occupancy must be 'counts' or 'valid', got {enc!r}")
    return enc


# ------------------------------------------------------------------------------------------- #
# Per-batch JAX augmentation
# ------------------------------------------------------------------------------------------- #


@register_augmentation("stream_summary_statistics")
def _stream_summary_statistics(params, rng, context=None):
    jax, jnp = _jax()
    obs_key = _sim_key(params)
    summary_key = str(params.get("summary_key", "sim_summary"))
    k_track = int(params.get("summary_track_bins", 10))
    k_vlos = int(params.get("summary_vlos_bins", 3))
    channels = dict(_DEFAULT_CHANNELS)
    channels.update(
        {k: int(v) for k, v in (params.get("summary_channels", {}) or {}).items() if k in channels}
    )
    ra_c, dec_c = channels["ra"], channels["dec"]
    par_c, mura_c, mudec_c, vlos_c = (
        channels["parallax"],
        channels["mu_ra"],
        channels["mu_dec"],
        channels["vlos"],
    )

    frames = _stream_frames(params, k_track, k_vlos, channels)
    scale, min_count, include_occupancy = _estimator_params(params)
    R_all = jnp.asarray(frames.R)  # (S, 3, 3)
    track_edges_all = jnp.asarray(frames.track_edges)  # (S, Kt+1)
    vlos_edges_all = jnp.asarray(frames.vlos_edges)  # (S, Kv+1)
    bins_t = jnp.arange(k_track)
    bins_v = jnp.arange(k_vlos)

    def _binned(vals, in_mask):
        return _binned_stats(jnp, vals, in_mask, scale, min_count)

    @jax.jit
    def _run(sim, attn, vmask, j):
        ra, dec = sim[..., ra_c], sim[..., dec_c]  # (n, P) degrees
        parallax, vlos = sim[..., par_c], sim[..., vlos_c]
        mura, mudec = sim[..., mura_c], sim[..., mudec_c]
        rar, decr = jnp.radians(ra), jnp.radians(dec)

        # positions -> stream frame
        n_vec = jnp.stack(
            [jnp.cos(decr) * jnp.cos(rar), jnp.cos(decr) * jnp.sin(rar), jnp.sin(decr)], -1
        )  # (n, P, 3)
        Rj = R_all[j]  # (n, 3, 3)
        npr = jnp.einsum("nij,npj->npi", Rj, n_vec)
        phi1 = jnp.degrees(jnp.arctan2(npr[..., 1], npr[..., 0]))
        phi2 = jnp.degrees(jnp.arcsin(jnp.clip(npr[..., 2], -1.0, 1.0)))

        # proper motions -> stream frame (rotate the ICRS tangent vector, then project)
        e = jnp.stack([-jnp.sin(rar), jnp.cos(rar), jnp.zeros_like(rar)], -1)
        m = jnp.stack(
            [-jnp.sin(decr) * jnp.cos(rar), -jnp.sin(decr) * jnp.sin(rar), jnp.cos(decr)], -1
        )
        v = mura[..., None] * e + mudec[..., None] * m  # (n, P, 3), μ_ra already ×cos(dec)
        vpr = jnp.einsum("nij,npj->npi", Rj, v)
        p1, p2 = jnp.radians(phi1), jnp.radians(phi2)
        ep = jnp.stack([-jnp.sin(p1), jnp.cos(p1), jnp.zeros_like(p1)], -1)
        mp = jnp.stack(
            [-jnp.sin(p2) * jnp.cos(p1), -jnp.sin(p2) * jnp.sin(p1), jnp.cos(p2)], -1
        )
        mu_phi1 = jnp.sum(vpr * ep, -1)
        mu_phi2 = jnp.sum(vpr * mp, -1)

        attended = attn.astype(bool)
        measured = vmask.astype(bool) & attended

        # per-row bin assignment via each stream's own quantile edges; out-of-range stars are
        # excluded rather than clipped into the end bins (see _bin_assignment)
        te, ve = track_edges_all[j], vlos_edges_all[j]  # (n, Kt+1), (n, Kv+1)
        idx_t, ok_t = _bin_assignment(jax, jnp, te, phi1, k_track)
        idx_v, ok_v = _bin_assignment(jax, jnp, ve, phi1, k_vlos)
        att_t = attended & ok_t
        meas_v = measured & ok_v
        in_t = (idx_t[:, None, :] == bins_t[None, :, None]) & att_t[:, None, :]
        in_v = (idx_v[:, None, :] == bins_v[None, :, None]) & meas_v[:, None, :]

        feats = []
        counts = []
        for q in (phi2, parallax, mu_phi1, mu_phi2):
            med, std, cnt = _binned(q, in_t)
            feats += [med, std]
        counts.append(cnt)  # identical for all four (same in_t mask)
        vmed, vstd, vcnt = _binned(vlos, in_v)
        feats += [vmed, vstd]
        counts.append(vcnt)

        n_att = jnp.clip(attended.sum(1), 1)
        frac_meas = measured.sum(1) / n_att
        frac_att = attended.sum(1) / attended.shape[1]
        # fraction of attended stars falling outside the real φ1 range — real extent information
        # that used to be silently folded into the end bins by clipping
        frac_out = jnp.where(attended & ~ok_t, 1.0, 0.0).sum(1) / n_att
        phi1_att = jnp.where(attended, phi1, jnp.nan)
        extent = jnp.nanmax(phi1_att, 1) - jnp.nanmin(phi1_att, 1)
        asym = (
            jnp.where(attended, phi1 > 0, False).sum(1) - jnp.where(attended, phi1 < 0, False).sum(1)
        ) / n_att
        scalars = jnp.stack([frac_meas, frac_att, extent, asym, frac_out], -1)

        # Stream index as an explicit feature so the summary MLP is stream-aware (like the
        # particle backbone's concatenate_stream_index channel). Standardized with the rest of
        # sim_summary at training time; for 3 streams the MLP resolves them cleanly.
        j_feat = j.reshape(-1, 1).astype(jnp.float32)

        blocks = feats + (counts if include_occupancy else []) + [scalars, j_feat]
        out = jnp.concatenate(blocks, axis=-1)
        # NaN -> 0 keeps the network's input finite; the occupancy channels above carry the
        # "this bin was empty / under-populated" information that the 0 would otherwise hide.
        return jnp.nan_to_num(out, nan=0.0).astype(jnp.float32)

    def aug(batch):
        sim = jnp.asarray(batch[obs_key])
        attn = jnp.asarray(batch["attention_mask"])[:, 0, :]
        vmask = jnp.asarray(batch["vlos_mask"])[:, 0, :]
        j = _stream_ids_jax(batch)
        batch[summary_key] = _run(sim, attn, vmask, j)
        return batch

    return aug


# ------------------------------------------------------------------------------------------- #
# φ1-gridded variant: one (median, std, j) triplet per observable per φ1 bin, laid out as a
# time series for a single TimeSeriesTransformer with φ1 as its ``time_axis``.
# ------------------------------------------------------------------------------------------- #

# The five stream-frame observables summarised per φ1 bin, in output order.
_GRID_OBSERVABLES = ("phi2", "parallax", "mu_phi1", "mu_phi2", "vlos")


@register_augmentation("stream_summary_grid")
def _stream_summary_grid(params, rng, context=None):
    """Per-stream summary statistics laid out as a **φ1 time series** for a single
    ``TimeSeriesTransformer`` (``time_axis=-1``).

    Unlike :func:`_stream_summary_statistics` (which flattens everything into one rank-2 vector),
    this writes ``batch["sim_summary"]`` as a rank-3 tensor ``(n, K_phi1, 5*3 + 1)``:

    * The **time axis** is φ1 — ``K_phi1 = summary_track_bins`` quantile bins (the same per-stream
      great-circle frame + edges as :func:`_stream_summary_statistics`). The φ1 bin-centre is the
      **last** channel, consumed by the network's ``time_axis=-1``.
    * At each φ1 bin, every one of the five observables (φ2, parallax, μ_φ1, μ_φ2, v_los) contributes
      the triplet ``[median, std, j]`` — the bin's median and std over the attended members and the
      stream index ``j`` (constant across bins of a given stream, so the Transformer is
      stream-aware). ``v_los`` uses **measured stars only** (native missing-v_los handling), on the
      SAME φ1 grid as the astrometric tracks.
    * Two **occupancy** channels (``n_track``, ``n_vlos``) give the number of members actually
      contributing to each bin. Statistics for empty / under-populated bins are still substituted
      with 0 so the network's input stays finite, but the occupancy channels mean that 0 is no
      longer indistinguishable from a genuine on-track, zero-dispersion measurement. Pair with the
      ``masked_time_series_transformer`` backbone, which reads them and masks those bins out.

    Channel layout (14 = 5 observables × [median, std] + n_track + n_vlos + j + φ1): since
    everything is flattened onto one feature axis, ``j`` (constant across a stream's bins) and the
    occupancy channels are carried **once** rather than repeated per observable, and the φ1
    bin-centre is last (``-1``):
    ``[med_φ2, std_φ2,  med_plx, std_plx,  med_μφ1, std_μφ1,  med_μφ2, std_μφ2,
       med_vlos, std_vlos,  n_track, n_vlos,  j,  φ1_centre]``.

    ``params.summary_include_std: false`` drops every per-bin std channel (the misspecification
    localization of 2026-07-15 found the real-vs-sim MMD flag lives in the dispersions —
    ``std_phi2`` above all), leaving the medians-only layout
    ``[med_φ2, med_plx, med_μφ1, med_μφ2, med_vlos, n_track, n_vlos, j, φ1_centre]`` (9 channels).
    Default true. ``params.summary_include_occupancy: false`` drops the two occupancy channels,
    restoring the pre-fix layouts (12 / 7 channels) for comparison runs.
    ``params.summary_occupancy: valid`` writes the occupancy channels as ±1 validity flags instead
    of counts — REQUIRED with the masked backbones whenever ``summary_variables`` are standardized
    (see :func:`occupancy_encoding`).

    Pair with a ``time_series_transformer`` summary backbone carrying ``params.time_axis: -1``.
    """
    jax, jnp = _jax()
    obs_key = _sim_key(params)
    summary_key = str(params.get("summary_key", "sim_summary"))
    k_track = int(params.get("summary_track_bins", 10))
    # vlos rides the SAME φ1 grid as the astrometric tracks here (shared time axis) — k_vlos is only
    # consulted by _stream_frames' cache key / vlos edges, which this variant does not use.
    k_vlos = int(params.get("summary_vlos_bins", 3))
    include_std = bool(params.get("summary_include_std", True))
    channels = dict(_DEFAULT_CHANNELS)
    channels.update(
        {k: int(v) for k, v in (params.get("summary_channels", {}) or {}).items() if k in channels}
    )
    ra_c, dec_c = channels["ra"], channels["dec"]
    par_c, mura_c, mudec_c, vlos_c = (
        channels["parallax"],
        channels["mu_ra"],
        channels["mu_dec"],
        channels["vlos"],
    )

    frames = _stream_frames(params, k_track, k_vlos, channels)
    scale, min_count, include_occupancy = _estimator_params(params)
    encoding = occupancy_encoding(params)
    R_all = jnp.asarray(frames.R)  # (S, 3, 3)
    track_edges_all = jnp.asarray(frames.track_edges)  # (S, Kt+1)
    bins_t = jnp.arange(k_track)

    def _binned(vals, in_mask):
        return _binned_stats(jnp, vals, in_mask, scale, min_count)

    @jax.jit
    def _run(sim, attn, vmask, j):
        ra, dec = sim[..., ra_c], sim[..., dec_c]  # (n, P) degrees
        parallax, vlos = sim[..., par_c], sim[..., vlos_c]
        mura, mudec = sim[..., mura_c], sim[..., mudec_c]
        rar, decr = jnp.radians(ra), jnp.radians(dec)

        # positions -> stream frame
        n_vec = jnp.stack(
            [jnp.cos(decr) * jnp.cos(rar), jnp.cos(decr) * jnp.sin(rar), jnp.sin(decr)], -1
        )  # (n, P, 3)
        Rj = R_all[j]  # (n, 3, 3)
        npr = jnp.einsum("nij,npj->npi", Rj, n_vec)
        phi1 = jnp.degrees(jnp.arctan2(npr[..., 1], npr[..., 0]))
        phi2 = jnp.degrees(jnp.arcsin(jnp.clip(npr[..., 2], -1.0, 1.0)))

        # proper motions -> stream frame (rotate the ICRS tangent vector, then project)
        e = jnp.stack([-jnp.sin(rar), jnp.cos(rar), jnp.zeros_like(rar)], -1)
        m = jnp.stack(
            [-jnp.sin(decr) * jnp.cos(rar), -jnp.sin(decr) * jnp.sin(rar), jnp.cos(decr)], -1
        )
        v = mura[..., None] * e + mudec[..., None] * m  # (n, P, 3), μ_ra already ×cos(dec)
        vpr = jnp.einsum("nij,npj->npi", Rj, v)
        p1, p2 = jnp.radians(phi1), jnp.radians(phi2)
        ep = jnp.stack([-jnp.sin(p1), jnp.cos(p1), jnp.zeros_like(p1)], -1)
        mp = jnp.stack(
            [-jnp.sin(p2) * jnp.cos(p1), -jnp.sin(p2) * jnp.sin(p1), jnp.cos(p2)], -1
        )
        mu_phi1 = jnp.sum(vpr * ep, -1)
        mu_phi2 = jnp.sum(vpr * mp, -1)

        attended = attn.astype(bool)
        measured = vmask.astype(bool) & attended

        # per-row φ1 bin assignment via each stream's own quantile track edges (shared by all obs);
        # out-of-range stars are excluded rather than clipped into the end bins
        te = track_edges_all[j]  # (n, Kt+1)
        idx_t, ok_t = _bin_assignment(jax, jnp, te, phi1, k_track)
        att_t = attended & ok_t
        in_t = (idx_t[:, None, :] == bins_t[None, :, None]) & att_t[:, None, :]  # (n, K, P)
        in_v = in_t & measured[:, None, :]  # v_los: same grid, measured stars only

        j_bc = jnp.broadcast_to(j.reshape(-1, 1).astype(jnp.float32), (j.shape[0], k_track))
        obs_masks = {
            "phi2": (phi2, in_t),
            "parallax": (parallax, in_t),
            "mu_phi1": (mu_phi1, in_t),
            "mu_phi2": (mu_phi2, in_t),
            "vlos": (vlos, in_v),
        }
        per_obs = []
        n_track = n_vlos = None
        for name in _GRID_OBSERVABLES:
            vals, mask = obs_masks[name]
            med, std, cnt = _binned(vals, mask)  # (n, K) each
            if name == "vlos":
                n_vlos = cnt
            else:
                n_track = cnt  # identical for all astrometric observables (same in_t mask)
            if include_std:
                per_obs.append(jnp.stack([med, std], axis=-1))  # (n, K, 2)
            else:
                per_obs.append(med[..., None])  # (n, K, 1) — medians only

        # Per-bin occupancy: the channels the masked backbone reads to tell an empty bin from a
        # genuinely cold one (and that make the NaN -> 0 substitution below non-deceptive).
        if encoding == "valid":  # sign-encoded validity: survives the approximator's standardization
            n_track = jnp.where(n_track >= min_count, 1.0, -1.0)
            n_vlos = jnp.where(n_vlos >= min_count, 1.0, -1.0)
        occ = [n_track[..., None], n_vlos[..., None]] if include_occupancy else []

        # j once (constant across a stream's bins) + φ1 bin-centre last (time_axis=-1)
        centre = 0.5 * (te[:, :-1] + te[:, 1:])  # (n, K)
        grid = jnp.concatenate(
            per_obs + occ + [j_bc[..., None], centre[..., None]], axis=-1
        )  # (n, K, 5*(2|1) + (2) + 2)
        return jnp.nan_to_num(grid, nan=0.0).astype(jnp.float32)

    def aug(batch):
        sim = jnp.asarray(batch[obs_key])
        attn = jnp.asarray(batch["attention_mask"])[:, 0, :]
        vmask = jnp.asarray(batch["vlos_mask"])[:, 0, :]
        j = _stream_ids_jax(batch)
        batch[summary_key] = _run(sim, attn, vmask, j)
        return batch

    return aug
