"""MaskedFusionNetwork.frozen_weights: a backbone is loaded from a pretrained npz and frozen."""

import numpy as np
from omegaconf import OmegaConf

from hydrabflow.config import SummaryNetworkConfig
from hydrabflow.registry import build_summary_network

SPEC = {"type": "time_series_transformer", "summary_dim": 4, "num_blocks": 1, "num_heads": 2,
        "embed_dim_per_head": 2, "dropout": 0.5}


def _cfg(frozen=None):
    return OmegaConf.merge(OmegaConf.structured(SummaryNetworkConfig), {
        "type": "fusion",
        "params": {"backbones": {"a": SPEC, "b": SPEC}, "frozen_weights": frozen or {}},
    })


def test_frozen_backbone_loads_weights_and_is_not_trainable(tmp_path):
    x = {"a": np.random.rand(3, 7, 2).astype("float32"), "b": np.random.rand(3, 5, 1).astype("float32")}

    # "pretrained" single-modality net on key b
    pre = build_summary_network(OmegaConf.merge(OmegaConf.structured(SummaryNetworkConfig), SPEC))
    pre.build(x["b"].shape)
    path = tmp_path / "w.npz"
    np.savez(path, *[np.asarray(w) for w in pre.get_weights()])

    net = build_summary_network(_cfg({"b": str(path)}), ["a", "b"])
    net.build({k: v.shape for k, v in x.items()})

    for w_net, w_pre in zip(net.backbones["b"].get_weights(), pre.get_weights()):
        np.testing.assert_array_equal(w_net, w_pre)
    frozen_ids = {id(v) for v in net.backbones["b"].weights}
    assert frozen_ids and not frozen_ids & {id(v) for v in net.trainable_weights}
    assert net.backbones["a"].trainable_weights

    # frozen branch ignores training=True (no dropout): its output slice is deterministic
    out1 = np.asarray(net(x, training=True))[:, 4:]
    out2 = np.asarray(net(x, training=True))[:, 4:]
    np.testing.assert_allclose(out1, out2)
    np.testing.assert_allclose(out1, np.asarray(pre(x["b"], training=False)), rtol=1e-5, atol=1e-6)


def test_warm_backbone_loads_weights_and_stays_trainable(tmp_path):
    x = {"a": np.random.rand(3, 7, 2).astype("float32"), "b": np.random.rand(3, 5, 1).astype("float32")}
    pre = build_summary_network(OmegaConf.merge(OmegaConf.structured(SummaryNetworkConfig), SPEC))
    pre.build(x["b"].shape)
    path = tmp_path / "w.npz"
    np.savez(path, *[np.asarray(w) for w in pre.get_weights()])

    cfg = _cfg()
    cfg.params.warm_weights = {"b": str(path)}
    net = build_summary_network(cfg, ["a", "b"])
    net.build({k: v.shape for k, v in x.items()})

    for w_net, w_pre in zip(net.backbones["b"].get_weights(), pre.get_weights()):
        np.testing.assert_array_equal(w_net, w_pre)
    warm_ids = {id(v) for v in net.backbones["b"].trainable_weights}
    assert warm_ids and warm_ids <= {id(v) for v in net.trainable_weights}


def test_post_mlp_keeps_branch_width_and_uses_extra_inputs():
    """params.post: MLP on [backbone(b), c] -> same width as b's summary_dim; c has no backbone."""
    import keras

    cfg = OmegaConf.merge(OmegaConf.structured(SummaryNetworkConfig), {
        "type": "fusion",
        "params": {"backbones": {"a": SPEC, "b": SPEC}, "post": {"b": {"inputs": ["c"], "widths": [8]}}},
    })
    rng = np.random.default_rng(0)
    x = {"a": rng.random((3, 7, 2), dtype="float32"), "b": rng.random((3, 5, 1), dtype="float32"),
         "c": rng.random((3, 3), dtype="float32")}
    net = build_summary_network(cfg, ["a", "b", "c"])
    out = np.asarray(net(x))
    assert out.shape == (3, 8)                      # 4 + 4: the post MLP keeps b's width
    y = dict(x, c=x["c"] + 1.0)
    assert np.allclose(np.asarray(net(y))[:, :4], out[:, :4])       # branch a untouched
    assert not np.allclose(np.asarray(net(y))[:, 4:], out[:, 4:])   # c reaches branch b
    clone = type(net).from_config(net.get_config())
    clone(x)
    assert set(clone.post) == {"b"} and clone.post_inputs == {"b": ["c"]}
    assert keras.ops.shape(clone(x)) == (3, 8)


def test_attach_and_noise_take_vectors():
    from hydrabflow.preprocessing.streams import AttachObservedSigmaZ
    from hydrabflow.registry import build_augmentations

    out = AttachObservedSigmaZ("pot_scalars", [2.0, -6.379, 1.17]).transform({"j": np.zeros((4, 1))})
    np.testing.assert_allclose(out["pot_scalars"], np.tile([2.0, -6.379, 1.17], (4, 1)))

    aug_cfg = OmegaConf.create({"steps": ["add_noise_to_sigma_z"],
                                "params": {"sigma_z_key": "pot_scalars", "sigma_z_err": [0.0, 1.0, 0.0]}})
    (aug,) = build_augmentations(aug_cfg, np.random.default_rng(0), context={})
    noisy = np.asarray(aug({"pot_scalars": out["pot_scalars"].astype("float32")})["pot_scalars"])
    np.testing.assert_allclose(noisy[:, [0, 2]], out["pot_scalars"][:, [0, 2]], rtol=1e-6)
    assert np.std(noisy[:, 1]) > 0
