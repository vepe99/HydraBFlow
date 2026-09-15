"""Stream-frame summary statistics: great-circle frame fit, stream-frame projection, the
``stream_summary_statistics`` augmentation (shape / no-NaN / stream-awareness / occupancy channels /
unbiased dispersion / out-of-range exclusion), the ``mlp`` and ``masked_time_series_transformer``
summary backbones, and config composition of the summaries-only / hybrid / masked-grid presets."""

from __future__ import annotations

import numpy as np
import pytest

from hydrabflow.augmentation.stream_summary import (
    _np_fit_frame,
    _np_phi1,
    _np_unit_vec,
)

K_TRACK, K_VLOS = 10, 3
# tracks + vlos + occupancy (n_track, n_vlos) + scalars (incl. out-of-range frac) + j
N_STATS = 4 * 2 * K_TRACK + 2 * K_VLOS + (K_TRACK + K_VLOS) + 5 + 1  # = 105
N_STATS_NO_OCC = 4 * 2 * K_TRACK + 2 * K_VLOS + 5 + 1  # occupancy channels dropped = 92
SIM_KEY = "sim_data_projected"


def _params(**extra):
    p = {
        "observable": SIM_KEY,
        "resources_dir": "assets/gaia",
        "target_streams": {"Pal5": 0, "NGC3201": 1, "M68": 2},
        "summary_track_bins": K_TRACK,
        "summary_vlos_bins": K_VLOS,
    }
    p.update(extra)
    return p


def _build(name, params, seed=0):
    from hydrabflow.registry import AUGMENTATIONS

    return AUGMENTATIONS.get(name)(params, np.random.default_rng(seed), {})


# --------------------------------------------------------------------------------------------- #
# Great-circle frame + projection
# --------------------------------------------------------------------------------------------- #


def test_fit_frame_is_proper_rotation_and_planar():
    """A synthetic great circle (φ2≈0 by construction) yields an orthonormal, det=+1 frame whose
    projection has a near-zero out-of-plane coordinate."""
    rng = np.random.default_rng(0)
    phi1 = np.sort(rng.uniform(-40, 40, size=200))
    # embed a thin great circle tilted off the equator, then rotate to sky coords
    theta = np.radians(phi1)
    pts = np.stack([np.cos(theta), np.sin(theta), 0.005 * rng.normal(size=theta.size)], -1)
    # arbitrary rotation into "sky"
    A = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    n = pts @ A.T
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ra = np.degrees(np.arctan2(n[:, 1], n[:, 0]))
    dec = np.degrees(np.arcsin(np.clip(n[:, 2], -1, 1)))

    R = _np_fit_frame(ra, dec)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
    npr = _np_unit_vec(ra, dec) @ R.T
    phi2 = np.degrees(np.arcsin(np.clip(npr[:, 2], -1, 1)))
    assert np.abs(phi2).max() < 3.0  # stream is thin in the fitted frame


def test_pm_rotation_preserves_speed():
    """Rotating the ICRS tangent velocity into the stream frame is a pure rotation → the on-sky
    proper-motion speed is invariant. Guards the e/m and ep/mp basis construction."""
    rng = np.random.default_rng(1)
    ra = rng.uniform(0, 360, 50)
    dec = rng.uniform(-80, 80, 50)
    mura, mudec = rng.normal(size=50), rng.normal(size=50)
    R = _np_fit_frame(ra, dec)

    rar, decr = np.radians(ra), np.radians(dec)
    e = np.stack([-np.sin(rar), np.cos(rar), np.zeros_like(rar)], -1)
    m = np.stack([-np.sin(decr) * np.cos(rar), -np.sin(decr) * np.sin(rar), np.cos(decr)], -1)
    v = mura[:, None] * e + mudec[:, None] * m
    vpr = v @ R.T
    npr = _np_unit_vec(ra, dec) @ R.T
    phi1 = np.arctan2(npr[:, 1], npr[:, 0])
    phi2 = np.arcsin(np.clip(npr[:, 2], -1, 1))
    ep = np.stack([-np.sin(phi1), np.cos(phi1), np.zeros_like(phi1)], -1)
    mp = np.stack([-np.sin(phi2) * np.cos(phi1), -np.sin(phi2) * np.sin(phi1), np.cos(phi2)], -1)
    mu_phi1 = np.sum(vpr * ep, 1)
    mu_phi2 = np.sum(vpr * mp, 1)

    np.testing.assert_allclose(
        np.hypot(mu_phi1, mu_phi2), np.hypot(mura, mudec), rtol=1e-6
    )


# --------------------------------------------------------------------------------------------- #
# The augmentation
# --------------------------------------------------------------------------------------------- #


def _toy_batch(n=9, particles=300, seed=0, zero_vlos_row=0):
    rng = np.random.default_rng(seed)
    sim = rng.normal(size=(n, particles, 6)).astype(np.float64)
    sim[..., 0] = rng.uniform(200, 250, size=(n, particles))
    sim[..., 1] = rng.uniform(-40, 40, size=(n, particles))
    attn = np.zeros((n, 1, particles), bool)
    vmask = np.zeros((n, 1, particles), bool)
    for i in range(n):
        k = rng.integers(60, 250)
        idx = rng.choice(particles, k, replace=False)
        attn[i, 0, idx] = True
        vmask[i, 0, rng.choice(idx, min(20, k), replace=False)] = True
    if zero_vlos_row is not None:
        vmask[zero_vlos_row, 0, :] = False  # a stream with no measured v_los at all
    j = np.array([[i % 3] for i in range(n)])
    return {
        SIM_KEY: sim,
        "attention_mask": attn.astype(float),
        "vlos_mask": vmask.astype(float),
        "j": j,
    }


def test_summary_shape_and_no_nan():
    aug = _build("stream_summary_statistics", _params())
    out = np.asarray(aug(_toy_batch())["sim_summary"])
    assert out.shape == (9, N_STATS)
    assert np.isfinite(out).all()  # incl. the zero-measured-vlos row


def test_summary_is_stream_aware():
    """The final feature is the stream index j (so the MLP can distinguish streams)."""
    batch = _toy_batch()
    out = np.asarray(_build("stream_summary_statistics", _params())(batch)["sim_summary"])
    np.testing.assert_array_equal(out[:, -1], np.asarray(batch["j"]).reshape(-1))


def test_unmeasured_vlos_does_not_corrupt_other_features():
    """A stream with zero measured v_los still yields finite track features (only its vlos-bin
    cells collapse to 0)."""
    out = np.asarray(_build("stream_summary_statistics", _params())(_toy_batch())["sim_summary"])
    track_block = out[0, : 4 * 2 * K_TRACK]
    assert np.isfinite(track_block).all()


def test_summary_grid_median_only_layout():
    """summary_include_std=false drops the per-bin std channels: (n, K, 14) -> (n, K, 9), with the
    median / occupancy / j / φ1 channels bit-identical to the full layout's."""
    batch_full, batch_med = _toy_batch(), _toy_batch()
    full = np.asarray(_build("stream_summary_grid", _params())(batch_full)["sim_summary"])
    med = np.asarray(
        _build("stream_summary_grid", _params(summary_include_std=False))(batch_med)["sim_summary"]
    )
    assert full.shape == (9, K_TRACK, 5 * 2 + 2 + 2)  # + n_track, n_vlos
    assert med.shape == (9, K_TRACK, 5 + 2 + 2)
    assert np.isfinite(med).all()
    # medians (even channels of the full layout) + trailing occupancy, j, φ1_centre are unchanged
    np.testing.assert_array_equal(med[..., :5], full[..., 0:10:2])
    np.testing.assert_array_equal(med[..., 5:], full[..., 10:])


def test_summary_grid_occupancy_channels_count_members():
    """The two occupancy channels hold the actual per-bin member counts, and dropping them via
    summary_include_occupancy restores the pre-fix channel count."""
    grid = np.asarray(_build("stream_summary_grid", _params())(_toy_batch())["sim_summary"])
    n_track, n_vlos = grid[..., 10], grid[..., 11]
    assert (n_track >= n_vlos).all()  # vlos uses a subset (measured stars only)
    assert (n_track >= 0).all() and n_track.sum() > 0
    # occupancy summed over bins never exceeds the attended count per row
    assert (n_track.sum(1) <= _toy_batch()["attention_mask"][:, 0, :].sum(1) + 1e-6).all()

    no_occ = np.asarray(
        _build("stream_summary_grid", _params(summary_include_occupancy=False))(_toy_batch())[
            "sim_summary"
        ]
    )
    assert no_occ.shape == (9, K_TRACK, 5 * 2 + 2)


def test_summary_occupancy_and_scale_options():
    """The flat variant exposes the same knobs: occupancy channels, and a MAD dispersion scale."""
    full = np.asarray(_build("stream_summary_statistics", _params())(_toy_batch())["sim_summary"])
    assert full.shape == (9, N_STATS)
    no_occ = np.asarray(
        _build("stream_summary_statistics", _params(summary_include_occupancy=False))(_toy_batch())[
            "sim_summary"
        ]
    )
    assert no_occ.shape == (9, N_STATS_NO_OCC)
    mad = np.asarray(
        _build("stream_summary_statistics", _params(summary_scale="mad"))(_toy_batch())[
            "sim_summary"
        ]
    )
    assert mad.shape == (9, N_STATS) and np.isfinite(mad).all()
    with pytest.raises(ValueError, match="summary_scale"):
        _build("stream_summary_statistics", _params(summary_scale="variance"))


def test_binned_stats_is_unbiased_and_flags_sparse_bins():
    """The estimator fix. With ddof=0 the population std is biased low by sqrt((N-1)/N) — 0.72x at
    N=3 and exactly 0 at N=1 — which is what made sparse (simulated) bins look cold. ddof=1 removes
    that, and bins below ``min_count`` return NaN instead of a fabricated 0."""
    import jax.numpy as jnp

    from hydrabflow.augmentation.stream_summary import _binned_stats

    rng = np.random.default_rng(0)
    n_rep, sigma = 4000, 1.0
    # bin 0 holds 3 members, bin 1 holds 1 — the two occupancy regimes that mattered
    vals = np.zeros((n_rep, 4))
    vals[:, :3] = rng.normal(0.0, sigma, (n_rep, 3))
    vals[:, 3] = rng.normal(0.0, sigma, n_rep)
    mask = np.zeros((n_rep, 2, 4), bool)
    mask[:, 0, :3] = True
    mask[:, 1, 3] = True

    _, disp, cnt = _binned_stats(jnp, jnp.asarray(vals), jnp.asarray(mask), "std", 3)
    disp, cnt = np.asarray(disp), np.asarray(cnt)
    np.testing.assert_array_equal(cnt[0], [3.0, 1.0])
    # ddof=1 is unbiased in variance; the mean of the std is within a few % for N=3
    assert abs(np.mean(disp[:, 0] ** 2) - sigma**2) < 0.05
    # the ddof=0 form would have been biased low by sqrt(2/3) = 0.816
    assert np.mean(disp[:, 0]) > 0.85 * sigma
    # a single-member bin cannot yield a dispersion at all
    assert np.isnan(disp[:, 1]).all()


def test_out_of_range_stars_are_excluded_not_clipped():
    """Stars beyond the outermost φ1 edge used to be clipped into the end bins, inflating their
    dispersion. They must now be dropped from the binned statistics entirely."""
    import jax
    import jax.numpy as jnp

    from hydrabflow.augmentation.stream_summary import _bin_assignment, _binned_stats

    edges = jnp.asarray([[0.0, 1.0, 2.0]])  # K = 2
    phi1 = jnp.asarray([[0.5, 0.6, 0.7, 1.5, 500.0]])  # last one far outside
    idx, in_range = _bin_assignment(jax, jnp, edges, phi1, 2)
    np.testing.assert_array_equal(np.asarray(in_range)[0], [True, True, True, True, False])

    attended = jnp.asarray([[True, True, True, True, True]])
    bins = jnp.arange(2)
    in_bin = (idx[:, None, :] == bins[None, :, None]) & (attended & in_range)[:, None, :]
    vals = jnp.asarray([[0.0, 0.0, 0.0, 10.0, 1.0e6]])
    _, disp, cnt = _binned_stats(jnp, vals, in_bin, "std", 2)
    # bin 1 keeps only the single legitimate member (the 1e6 outlier is gone), so no dispersion
    np.testing.assert_array_equal(np.asarray(cnt)[0], [3.0, 1.0])
    assert np.isnan(np.asarray(disp)[0, 1])


# --------------------------------------------------------------------------------------------- #
# masked_time_series_transformer backbone
# --------------------------------------------------------------------------------------------- #


def _masked_tst(summary_dim=8, min_count=3):
    from omegaconf import OmegaConf

    from hydrabflow.config import SummaryNetworkConfig
    from hydrabflow.registry import build_summary_network

    cfg = OmegaConf.merge(
        OmegaConf.structured(SummaryNetworkConfig),
        {
            "type": "masked_time_series_transformer",
            "summary_dim": summary_dim,
            "num_blocks": 1,
            "num_heads": 2,
            "params": {
                "time_axis": -1,
                "count_channel": 10,
                "vlos_count_channel": 11,
                "stat_channels": [0, 1, 2, 3, 4, 5, 6, 7],
                "vlos_stat_channels": [8, 9],
                "min_count": min_count,
            },
        },
    )
    return build_summary_network(cfg)


def _grid_batch(n=4, k=10, seed=0):
    """A (n, K, 14) stream_summary_grid-shaped batch with every bin well populated."""
    x = np.random.default_rng(seed).normal(0.0, 1.0, (n, k, 14)).astype("float32")
    x[..., 10] = 8.0  # n_track
    x[..., 11] = 5.0  # n_vlos
    x[..., 12] = 0.0  # j
    x[..., 13] = np.arange(k)  # φ1 bin centres
    return x


def test_masked_tst_ignores_empty_bin_content():
    """Whatever sits in the statistic channels of an under-populated bin must not reach the
    summary — this is what stops a substituted 0 from reading as an on-track measurement."""
    net = _masked_tst()
    x = _grid_batch()
    net(x)  # build

    a, b = x.copy(), x.copy()
    for arr, junk in ((a, 1.0e4), (b, -7.0e3)):
        arr[:, 7:, 10] = 0.0  # bins 7..9 empty
        arr[:, 7:, 11] = 0.0
        arr[:, 7:, 0:10] = junk
    out_a, out_b = np.asarray(net(a)), np.asarray(net(b))
    assert np.max(np.abs(out_a - out_b)) < 1e-5 * max(np.max(np.abs(out_a)), 1.0)


def _masked_mlp(summary_dim=8, min_count=3):
    from omegaconf import OmegaConf

    from hydrabflow.config import SummaryNetworkConfig
    from hydrabflow.registry import build_summary_network

    cfg = OmegaConf.merge(
        OmegaConf.structured(SummaryNetworkConfig),
        {
            "type": "masked_mlp",
            "summary_dim": summary_dim,
            "mlp_depth": 2,
            "mlp_width": 16,
            "params": {
                "count_channel": 10,
                "vlos_count_channel": 11,
                "stat_channels": [0, 1, 2, 3, 4, 5, 6, 7],
                "vlos_stat_channels": [8, 9],
                "min_count": min_count,
            },
        },
    )
    return build_summary_network(cfg)


def test_masked_mlp_shares_the_occupancy_prep():
    """The MLP ablation arm must hide exactly what the transformer arm hides: the statistics of
    under-populated bins, and the count magnitudes themselves."""
    net = _masked_mlp()
    x = _grid_batch()
    out = np.asarray(net(x))
    assert out.shape == (x.shape[0], 8)

    junk_a, junk_b = x.copy(), x.copy()
    for arr, junk in ((junk_a, 1.0e4), (junk_b, -7.0e3)):
        arr[:, 7:, 10] = 0.0  # bins 7..9 under-populated
        arr[:, 7:, 11] = 0.0
        arr[:, 7:, 0:10] = junk
    np.testing.assert_allclose(np.asarray(net(junk_a)), np.asarray(net(junk_b)), atol=1e-5)

    counts_a, counts_b = x.copy(), x.copy()
    counts_a[..., 10], counts_a[..., 11] = 8.0, 5.0
    counts_b[..., 10], counts_b[..., 11] = 400.0, 91.0
    np.testing.assert_allclose(np.asarray(net(counts_a)), np.asarray(net(counts_b)), atol=1e-6)


def test_masked_tst_never_sees_count_values():
    """The occupancy channels are a validity signal, not an observable: once they have decided
    which bins are valid, their magnitudes must not reach the transformer. Otherwise the network
    trains on a simulated bin's member count, which reflects the progenitor draw rather than
    anything the real catalogue measures."""
    net = _masked_tst(min_count=3)
    x = _grid_batch()
    net(x)  # build

    a, b = x.copy(), x.copy()
    a[..., 10], a[..., 11] = 8.0, 5.0        # both well above min_count ...
    b[..., 10], b[..., 11] = 400.0, 91.0     # ... so the validity masks are identical
    out_a, out_b = np.asarray(net(a)), np.asarray(net(b))
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)


def test_masked_tst_padding_equals_true_subset():
    """The property BayesFlow's own nets do NOT have: padding a K-bin grid out with empty bins and
    masking them gives the same summary as feeding only the populated bins.

    ``TimeSeriesTransformer.call`` ends with an unmasked ``self.pooling(inp)`` (and
    ``SetTransformer.call`` with an unmasked ``pooling_by_attention``), so masked positions still
    leak into the stock summary. This is the regression that proves the wrapper's masked pooling
    works, and it is what makes per-stream adaptive φ1 bin counts possible."""
    net = _masked_tst()
    k_full, k_valid = 10, 7
    x = _grid_batch(k=k_full, seed=1)
    net(x)

    padded = x.copy()
    padded[:, k_valid:, 10] = 0.0
    padded[:, k_valid:, 11] = 0.0
    padded[:, k_valid:, 0:10] = 123.0

    got = np.asarray(net(padded))
    ref = np.asarray(net(x[:, :k_valid, :]))
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_masked_tst_all_bins_empty_stays_finite():
    """A row whose every bin is under-populated must not produce 0/0."""
    net = _masked_tst()
    x = _grid_batch()
    net(x)
    empty = x.copy()
    empty[..., 10] = 0.0
    empty[..., 11] = 0.0
    assert np.isfinite(np.asarray(net(empty))).all()


def test_masked_tst_rejects_out_of_range_channels():
    net = _masked_tst()
    with pytest.raises(ValueError, match="out of range"):
        net(np.zeros((2, 5, 6), dtype="float32"))


def test_masked_tst_serialization_round_trip():
    import keras

    net = _masked_tst()
    x = _grid_batch()
    before = np.asarray(net(x))
    revived = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(net))
    revived.build(x.shape)
    revived.set_weights(net.get_weights())
    np.testing.assert_allclose(before, np.asarray(revived(x)), atol=1e-6)


# --------------------------------------------------------------------------------------------- #
# mlp summary backbone
# --------------------------------------------------------------------------------------------- #


def test_mlp_backbone_forward():
    from omegaconf import OmegaConf

    from hydrabflow.config import SummaryNetworkConfig
    from hydrabflow.registry import build_summary_network

    cfg = OmegaConf.merge(
        OmegaConf.structured(SummaryNetworkConfig),
        {"type": "mlp", "summary_dim": 16, "mlp_depth": 2, "mlp_width": 32},
    )
    net = build_summary_network(cfg)
    x = np.random.default_rng(0).normal(size=(8, N_STATS)).astype("float32")
    out = np.asarray(net(x))
    assert out.shape == (8, 16)


def test_feature_transformer_backbone_forward():
    """The Transformer alternative to `mlp` on the flat summary vector: reshapes (n, F) -> (n, F, 1)
    feature tokens and runs a TimeSeriesTransformer, returning (n, summary_dim)."""
    from omegaconf import OmegaConf

    from hydrabflow.config import SummaryNetworkConfig
    from hydrabflow.registry import build_summary_network

    cfg = OmegaConf.merge(
        OmegaConf.structured(SummaryNetworkConfig),
        {
            "type": "feature_transformer",
            "summary_dim": 16,
            "num_blocks": 2,
            "num_heads": 4,
            "params": {"embed_dim_multiplier": 4},
        },
    )
    net = build_summary_network(cfg)
    x = np.random.default_rng(0).normal(size=(8, N_STATS)).astype("float32")
    out = np.asarray(net(x))
    assert out.shape == (8, 16)


# --------------------------------------------------------------------------------------------- #
# Config composition + adapter drop retention
# --------------------------------------------------------------------------------------------- #

SUMSTATS_OVERRIDES = [
    "simulator=stream_agama_rnbody_huang",
    "composition=global",
    "augmentation=stream_global_sumstats",
    "preprocessing=stream_global_log10_sumstats",
]


def test_compose_hybrid(compose):
    cfg = compose(
        SUMSTATS_OVERRIDES
        + ["model=stream_fusion_model5_sumstats_hybrid", "adapter=stream_sumstats_hybrid"]
    )
    assert list(cfg.adapter.summary_variables) == ["sim_data_projected", "sim_summary", "vcirc_kms"]
    assert list(cfg.adapter.inference_variables)  # derived from the simulator
    assert cfg.model.summary_network.type == "fusion"


def test_compose_summaries_only_and_drop_retained(compose):
    from hydrabflow.pipeline.adapter import adapter_keys

    cfg = compose(
        SUMSTATS_OVERRIDES
        + ["model=stream_fusion_model5_sumstats_only", "adapter=stream_sumstats_only"]
    )
    assert list(cfg.adapter.summary_variables) == ["sim_summary", "vcirc_kms"]
    assert list(cfg.adapter.drop) == ["sim_data_projected"]
    # adapter_keys must retain the dropped raw star cloud so the summary augmentation can read it.
    assert "sim_data_projected" in adapter_keys(cfg)


def test_compose_masked_grid_model_and_build(compose):
    """The masked-grid preset composes and its fusion network builds/runs on the 14-channel grid."""
    from hydrabflow.registry import build_summary_network

    cfg = compose(
        [
            "simulator=stream_agama_rnbody_ibata_onedisk_beta3_m200c",
            "model=stream_fusion_ibata_grid_masked",
            "adapter=stream_ibata_sumstats",
            "augmentation=stream_global_ibata_grid",
            "preprocessing=stream_global_log10_ibata_sumstats",
            "composition=global",
        ]
    )
    assert cfg.model.summary_network.type == "fusion"
    backbones = cfg.model.summary_network.params.backbones
    assert backbones.sim_summary.type == "masked_time_series_transformer"

    net = build_summary_network(cfg.model.summary_network)
    rng = np.random.default_rng(0)
    grid = rng.normal(0.0, 1.0, (4, K_TRACK, 14)).astype("float32")
    grid[..., 10] = 8.0  # n_track
    grid[..., 11] = 5.0  # n_vlos
    grid[..., 13] = np.arange(K_TRACK)  # φ1 centres
    out = net(
        {
            "sim_summary": grid,
            "vcirc_kms": rng.normal(0.0, 1.0, (4, 50, 1)).astype("float32"),
            "vterm_kms": rng.normal(0.0, 1.0, (4, 19, 1)).astype("float32"),
        }
    )
    assert out.shape == (4, cfg.model.summary_network.params.head.output_dim)


# --------------------------------------------------------------------------------------------- #
# Member contamination
# --------------------------------------------------------------------------------------------- #


def _contam_params(**extra):
    """Real augmentation params (the contamination step needs the full StreamResources tables)."""
    from omegaconf import OmegaConf

    from conftest import compose_cfg

    cfg = compose_cfg(
        [
            "simulator=stream_agama_rnbody_ibata_m200c_v2",
            "augmentation=stream_global_ibata_grid_v2",
        ]
    )
    p = OmegaConf.to_container(cfg.augmentation.params, resolve=True)
    p["resources_dir"] = "assets/gaia"
    p.update(extra)
    return p


def _contam_batch(n=6, particles=300, attended=200, seed=0):
    rng = np.random.default_rng(seed)
    sim = np.zeros((n, particles, 6), dtype=np.float32)
    sim[..., 0] = rng.uniform(225.0, 245.0, (n, particles))
    sim[..., 1] = rng.uniform(-5.0, 5.0, (n, particles))
    sim[..., 2] = rng.normal(0.05, 0.002, (n, particles))
    sim[..., 3] = rng.normal(-2.7, 0.1, (n, particles))
    sim[..., 4] = rng.normal(-2.6, 0.1, (n, particles))
    sim[..., 5] = rng.normal(-58.0, 2.0, (n, particles))
    mask = np.zeros((n, 1, particles), dtype=np.float32)
    mask[:, 0, :attended] = 1.0
    return {
        SIM_KEY: sim,
        "attention_mask": mask,
        "vlos_mask": mask.copy(),
        "j": np.array([[i % 3] for i in range(n)]),
    }


def test_contamination_zero_fraction_is_a_no_op():
    """The step must be safe to leave in a chain: contamination_max_frac=0 changes nothing."""
    ref = _contam_batch()[SIM_KEY].copy()
    out = _build("contaminate_members", _contam_params(contamination_max_frac=0.0))(_contam_batch())
    np.testing.assert_allclose(np.asarray(out[SIM_KEY]), ref, atol=1e-5)


def test_contamination_replaces_only_attended_members_and_inflates_dispersion():
    base = _contam_batch()
    ref = base[SIM_KEY].copy()
    attended = base["attention_mask"][:, 0, :] > 0
    out = np.asarray(
        _build(
            "contaminate_members",
            _contam_params(contamination_max_frac=0.3, contamination_inflate=3.0),
        )(_contam_batch())[SIM_KEY]
    )

    changed = (np.abs(out - ref) > 1e-4).any(-1)
    # only selected members become interlopers, and at most the prior's upper bound of them
    assert not changed[~attended].any()
    assert 0.0 < changed[attended].mean() <= 0.3
    # the point of the step: the member sample's kinematic dispersion goes UP
    for ch in (2, 3, 4, 5):
        assert out[attended][:, ch].std() > ref[attended][:, ch].std()


def test_compose_v2_presets(compose):
    """The v2 simulator / augmentation / model presets compose together, and the estimator settings
    of the sim and real augmentation chains agree (otherwise the summaries are not comparable)."""
    cfg = compose(
        [
            "simulator=stream_agama_rnbody_ibata_m200c_v2",
            "model=stream_fusion_ibata_grid_masked",
            "adapter=stream_ibata_sumstats",
            "augmentation=stream_global_ibata_grid_v2",
            "preprocessing=stream_global_log10_ibata_sumstats",
            "composition=global",
        ]
    )
    # the step is present but DISABLED: the training set is trained without contamination, and at
    # max_frac 0 the step is an exact no-op (see test_contamination_zero_fraction_is_a_no_op), so it
    # can be switched on later with a single override and no new config
    assert "contaminate_members" in list(cfg.augmentation.steps)
    assert cfg.augmentation.params.contamination_max_frac == 0.0
    assert cfg.augmentation.params.summary_scale == "mad"
    # the masked backbone's min_count must match the augmentation's
    backbone = cfg.model.summary_network.params.backbones.sim_summary
    assert backbone.params.min_count == cfg.augmentation.params.summary_min_count

    real = compose(
        [
            "simulator=stream_agama_rnbody_ibata_m200c_v2",
            "augmentation=stream_real_global_ibata_grid_v2",
        ]
    )
    # real data must NOT be contaminated a second time, but must share the estimator settings
    assert "contaminate_members" not in list(real.augmentation.steps)
    assert real.augmentation.params.summary_scale == cfg.augmentation.params.summary_scale
    assert real.augmentation.params.summary_min_count == cfg.augmentation.params.summary_min_count


def test_v2_simulator_frees_the_intended_parameters(compose):
    """v2 drops the rotation-curve rejection prior and promotes the previously-pinned halo/bulge
    knobs and the stripping age to inferred parameters."""
    from hydrabflow.registry import get_simulator

    sim = get_simulator(compose(["simulator=stream_agama_rnbody_ibata_m200c_v2"]).simulator)
    assert sim._vcirc_rejection is None
    for name in (
        "alpha_TwoPowerTriaxial_halo",
        "p_TwoPowerTriaxial_halo",
        "tilt_TwoPowerTriaxial_halo",
        "rho_Bulge",
    ):
        assert name in sim.global_parameter_names
    # t_end freed for the FIRST stream too, so it really is an inferred local (see
    # stream_agama.local_parameter_names, which reads the first stream only)
    assert "t_end" in sim.local_parameter_names
    assert "m_progenitor" in sim.local_parameter_names
