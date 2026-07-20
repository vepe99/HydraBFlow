"""Stream-frame summary statistics: great-circle frame fit, stream-frame projection, the
``stream_summary_statistics`` augmentation (shape / no-NaN / stream-awareness), the ``mlp`` summary
backbone, and config composition of the summaries-only / hybrid presets."""

from __future__ import annotations

import numpy as np
import pytest

from hydrabflow.augmentation.stream_summary import (
    _np_fit_frame,
    _np_phi1,
    _np_unit_vec,
)

K_TRACK, K_VLOS = 10, 3
N_STATS = 4 * 2 * K_TRACK + 2 * K_VLOS + 4 + 1  # tracks + vlos + scalars + j  = 91
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
    from hydrabflow.augmentation.registry import _REGISTRY

    return _REGISTRY[name](params, np.random.default_rng(seed))


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
    """summary_include_std=false drops the per-bin std channels: (n, K, 12) -> (n, K, 7), with the
    median / j / φ1 channels bit-identical to the full layout's."""
    batch_full, batch_med = _toy_batch(), _toy_batch()
    full = np.asarray(_build("stream_summary_grid", _params())(batch_full)["sim_summary"])
    med = np.asarray(
        _build("stream_summary_grid", _params(summary_include_std=False))(batch_med)["sim_summary"]
    )
    assert full.shape == (9, K_TRACK, 5 * 2 + 2)
    assert med.shape == (9, K_TRACK, 5 + 2)
    assert np.isfinite(med).all()
    # medians (even channels of the full layout) + trailing j, φ1_centre are unchanged
    np.testing.assert_array_equal(med[..., :5], full[..., 0:10:2])
    np.testing.assert_array_equal(med[..., 5:], full[..., 10:])


# --------------------------------------------------------------------------------------------- #
# mlp summary backbone
# --------------------------------------------------------------------------------------------- #


def test_mlp_backbone_forward():
    from omegaconf import OmegaConf

    from hydrabflow.config.schema import SummaryNetworkConfig
    from hydrabflow.networks.factory import build_summary_network

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

    from hydrabflow.config.schema import SummaryNetworkConfig
    from hydrabflow.networks.factory import build_summary_network

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
