"""The single-modality coupling-flow presets compose and build."""

import pytest

from tests.conftest import compose_cfg


@pytest.mark.parametrize(
    "model, adapter, aug, summary_key",
    [
        ("imm_vcirc", "stream_vcirc_only", "stream_global_vcirc_only", "vcirc_kms"),
        ("imm_vcirc_mlp", "stream_vcirc_only", "stream_global_vcirc_only", "vcirc_kms"),
        ("imm_streams_mlp", "stream_streams_only", "stream_global_sumstats_only", "sim_summary"),
        ("imm_streams", "stream_streams_only", "stream_global_sumstats_only", "sim_summary"),
    ],
)
def test_imm_presets_compose_and_build(model, adapter, aug, summary_key):
    pytest.importorskip("agama")
    from hydrabflow.pipeline.workflow import build_workflow

    cfg = compose_cfg(
        [
            "simulator=stream_agama_spray_massloss_ibata_m200c_v4_palau",
            f"model={model}", f"adapter={adapter}", f"augmentation={aug}",
            "preprocessing=stream_global_log10_ibata_sumstats", "composition=global",
        ]
    )
    assert list(cfg.adapter.summary_variables) == [summary_key]
    wf = build_workflow(cfg)
    assert type(wf.approximator.inference_network).__name__ == "CouplingFlow"


def test_mlp_standalone_is_a_summary_network():
    """As the only summary net, BayesFlow calls compute_metrics(stage=...) -- a bare Sequential
    cannot take it (the imm_vcirc_mlp crash)."""
    import keras
    from omegaconf import OmegaConf

    from hydrabflow.registry import SUMMARY_NETWORKS

    cfg = OmegaConf.create({"summary_dim": 8, "mlp_depth": 2, "mlp_width": 16, "dropout": 0.0})
    net = SUMMARY_NETWORKS.get("mlp")(cfg)
    x = keras.ops.ones((4, 19, 1))
    out = net.compute_metrics(x, stage="training")["outputs"]
    assert tuple(out.shape) == (4, 8)
    clone = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(net))
    assert type(clone).__name__ == type(net).__name__
