"""Missing-v_los handling: fill modes (mask_vlos / impute_vlos) and the missingness-aware
masked_set_transformer summary network (zeroed channels + learned missing embedding)."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

SIM_KEY = "sim_data_projected"
N_PARTICLES = 24
VLOS_PER_STREAM = {"Pal5": 5, "NGC3201": 7, "M68": 9}


def _stream_params(compose, **extra):
    """The real stream_global params (resources from the git-tracked assets/gaia copy), with a
    small per-stream v_los budget so a 24-particle toy batch has both measured and missing."""
    cfg = compose(
        [
            "simulator=stream_agama",
            "model=stream_fusion",
            "adapter=stream",
            "augmentation=stream_global",
            "augmentation.params.resources_dir=assets/gaia",
        ]
    )
    params = OmegaConf.to_container(cfg.augmentation.params, resolve=True)
    params["min_star_with_vlos"] = dict(VLOS_PER_STREAM)
    params.update(extra)
    return params


def _vlos_batch(n=4, particles=N_PARTICLES, seed=0):
    rng = np.random.default_rng(seed)
    sim = rng.normal(size=(n, particles, 6)).astype(np.float32)
    sim[:, :, -1] = rng.normal(loc=100.0, scale=10.0, size=(n, particles))  # far from 0
    sigma = np.abs(rng.normal(size=(n, particles, 6))).astype(np.float32) + 0.1
    return {
        SIM_KEY: sim,
        "sigma_errors": sigma,
        "attention_mask": np.ones((n, 1, particles), dtype=bool),
        "j": np.array([[0], [1], [2], [0]]),
    }


def _build(name, params, seed=0):
    from hydrabflow.augmentation.registry import _REGISTRY

    return _REGISTRY[name](params, np.random.default_rng(seed))


# --------------------------------------------------------------------------------------------- #
# mask_vlos fill modes
# --------------------------------------------------------------------------------------------- #


def test_mask_vlos_mean_mode_unchanged(compose):
    """Default (mean) mode: unmeasured v_los carries the mean of the measured stars, sigma
    their sample std — the historical behavior."""
    batch = _vlos_batch()
    out = _build("mask_vlos", _stream_params(compose))(dict(batch))

    vlos_mask = np.asarray(out["vlos_mask"])[:, 0, :].astype(bool)
    sim, sigma = np.asarray(out[SIM_KEY]), np.asarray(out["sigma_errors"])
    assert 0 < vlos_mask.sum() < vlos_mask.size

    for i in range(sim.shape[0]):
        measured = batch[SIM_KEY][i, vlos_mask[i], -1]
        np.testing.assert_allclose(sim[i, ~vlos_mask[i], -1], measured.mean(), rtol=1e-5)
        np.testing.assert_allclose(sigma[i, ~vlos_mask[i], -1], measured.std(), rtol=1e-4)
        np.testing.assert_allclose(sim[i, vlos_mask[i], -1], measured, rtol=1e-6)


def test_mask_vlos_zero_mode(compose):
    batch = _vlos_batch()
    out = _build("mask_vlos", _stream_params(compose, vlos_impute="zero"))(dict(batch))

    vlos_mask = np.asarray(out["vlos_mask"])[:, 0, :].astype(bool)
    sim, sigma = np.asarray(out[SIM_KEY]), np.asarray(out["sigma_errors"])

    np.testing.assert_array_equal(sim[..., -1][~vlos_mask], 0.0)
    np.testing.assert_array_equal(sigma[..., -1][~vlos_mask], 0.0)
    # Measured stars and the other channels are untouched.
    np.testing.assert_allclose(sim[..., -1][vlos_mask], batch[SIM_KEY][..., -1][vlos_mask])
    np.testing.assert_allclose(sim[..., :-1], batch[SIM_KEY][..., :-1])


def test_vlos_impute_validated(compose):
    with pytest.raises(ValueError, match="vlos_impute"):
        _build("impute_vlos", {"vlos_impute": "median"})


# --------------------------------------------------------------------------------------------- #
# impute_vlos (real-data chain)
# --------------------------------------------------------------------------------------------- #


def _real_like_batch():
    """Batch shaped like the real path: vlos_mask given, unmeasured v_los pre-filled with the
    measured-star mean (exactly how the shipped Gaia npz is built)."""
    batch = _vlos_batch(seed=3)
    n, particles = batch[SIM_KEY].shape[:2]
    rng = np.random.default_rng(7)
    vlos_mask = rng.random((n, particles)) < 0.3
    vlos_mask[:, 0] = True  # at least one measured star per row
    for i in range(n):
        batch[SIM_KEY][i, ~vlos_mask[i], -1] = batch[SIM_KEY][i, vlos_mask[i], -1].mean()
    batch["vlos_mask"] = vlos_mask[:, None, :].astype(np.float64)
    return batch, vlos_mask


def test_impute_vlos_mean_is_noop_on_real_fill():
    batch, _ = _real_like_batch()
    out = _build("impute_vlos", {})(dict(batch))
    np.testing.assert_allclose(np.asarray(out[SIM_KEY]), batch[SIM_KEY], rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.asarray(out["sigma_errors"]), batch["sigma_errors"])


def test_impute_vlos_zero_mode():
    batch, vlos_mask = _real_like_batch()
    out = _build("impute_vlos", {"vlos_impute": "zero"})(dict(batch))
    sim, sigma = np.asarray(out[SIM_KEY]), np.asarray(out["sigma_errors"])

    np.testing.assert_array_equal(sim[..., -1][~vlos_mask], 0.0)
    np.testing.assert_array_equal(sigma[..., -1][~vlos_mask], 0.0)
    np.testing.assert_allclose(sim[..., -1][vlos_mask], batch[SIM_KEY][..., -1][vlos_mask])
    np.testing.assert_allclose(sigma[..., -1][vlos_mask], batch["sigma_errors"][..., -1][vlos_mask])
    np.testing.assert_allclose(sim[..., :-1], batch[SIM_KEY][..., :-1])


def test_real_global_chain_includes_impute_vlos(compose):
    cfg = compose(
        [
            "simulator=stream_agama", "model=stream_fusion", "adapter=stream",
            "composition=global", "preprocessing=stream_real_global",
            "augmentation=stream_real_global",
        ]
    )
    steps = list(cfg.augmentation.steps)
    assert "impute_vlos" in steps
    assert steps.index("impute_vlos") > steps.index("sample_obs_error")


# --------------------------------------------------------------------------------------------- #
# masked_set_transformer
# --------------------------------------------------------------------------------------------- #

N_CHANNELS = 15  # stream_global layout: 0-5 obs (vlos=5), 6-11 sigma (11), 12 mag, 13 mask, 14 j
VALUE_CH, SIGMA_CH, MASK_CH = 5, 11, 13


def _tiny_masked_net():
    from hydrabflow.config.schema import SummaryNetworkConfig
    from hydrabflow.networks.factory import build_summary_network

    spec = {
        "type": "masked_set_transformer",
        "summary_dim": 4,
        "num_blocks": 1,
        "num_heads": 2,
        "embed_dim": 8,
        "mlp_depth": 1,
        "mlp_width": 8,
        "dropout": 0.0,
        "params": {
            "value_channels": [VALUE_CH],
            "sigma_channels": [SIGMA_CH],
            "mask_channel": MASK_CH,
        },
    }
    return build_summary_network(OmegaConf.merge(OmegaConf.structured(SummaryNetworkConfig), spec))


def _star_batch(seed=0, n=2, particles=12):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, particles, N_CHANNELS)).astype(np.float32)
    mask = (rng.random((n, particles)) < 0.5).astype(np.float32)
    mask[:, 0], mask[:, 1] = 1.0, 0.0  # both populations present
    x[..., MASK_CH] = mask
    return x, mask.astype(bool)


def test_masked_set_transformer_registered():
    import hydrabflow.networks  # noqa: F401  (discovery)
    from hydrabflow.networks.factory import _SUMMARY_BUILDERS

    assert "masked_set_transformer" in _SUMMARY_BUILDERS


def test_masked_net_output_shape_and_missing_invariance():
    net = _tiny_masked_net()
    x, measured = _star_batch()
    out = np.asarray(net(x))
    assert out.shape == (2, 4)

    # Changing v_los value/sigma of UNMEASURED stars must not change the summary...
    x_poked = x.copy()
    x_poked[..., VALUE_CH][~measured] += 1e3
    x_poked[..., SIGMA_CH][~measured] += 1e2
    np.testing.assert_allclose(np.asarray(net(x_poked)), out, rtol=1e-5, atol=1e-5)

    # ...while changing a MEASURED star's v_los must.
    x_meas = x.copy()
    x_meas[..., VALUE_CH][measured] += 10.0
    assert not np.allclose(np.asarray(net(x_meas)), out, rtol=1e-5, atol=1e-5)


def test_masked_net_missing_embedding_is_used():
    net = _tiny_masked_net()
    x, _ = _star_batch(seed=1)
    x[..., VALUE_CH] = 0.0
    x[..., SIGMA_CH] = 0.0  # remove the channel-zeroing pathway: only the token can differ
    net(x)  # build
    net.missing_token.assign(np.ones(net.embed_dim, dtype=np.float32))

    x_flipped = x.copy()
    x_flipped[..., MASK_CH] = 1.0 - x_flipped[..., MASK_CH]
    assert not np.allclose(np.asarray(net(x_flipped)), np.asarray(net(x)), rtol=1e-5, atol=1e-5)


def test_masked_net_bad_channel_indices_raise():
    import keras

    net = _tiny_masked_net()
    with pytest.raises(ValueError, match="out of range"):
        net(keras.ops.zeros((2, 12, 8)))  # 8 channels < mask_channel 13


def test_masked_net_serialization_roundtrip():
    import keras

    net = _tiny_masked_net()
    x, _ = _star_batch(seed=2)
    out = np.asarray(net(x))

    clone = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(net))
    clone(x)  # build
    clone.set_weights(net.get_weights())
    np.testing.assert_allclose(np.asarray(clone(x)), out, rtol=1e-5, atol=1e-5)
    assert clone.mask_channel == MASK_CH and clone.value_channels == [VALUE_CH]


def test_maskedvlos_model_config_composes(compose):
    cfg = compose(
        [
            "simulator=stream_agama", "model=stream_fusion_model5_maskedvlos",
            "adapter=stream", "composition=global",
        ]
    )
    backbone = cfg.model.summary_network.params["backbones"][SIM_KEY]
    assert backbone["type"] == "masked_set_transformer"
    assert backbone["params"]["mask_channel"] == MASK_CH
    assert cfg.model.summary_network.params["mask_backbone"] == SIM_KEY
