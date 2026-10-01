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
