"""Config composition + schema validation smoke tests."""

from __future__ import annotations

from omegaconf import OmegaConf


def test_root_config_composes(cfg):
    assert cfg.seed == 42
    assert cfg.simulator.name == "two_moons"  # shipped default: works out of the box
    assert cfg.model.name == "default"
    assert cfg.model.summary_network.type == "set_transformer"
    assert cfg.model.inference_network.type == "flow_matching"


def test_interpolations_resolve(cfg):
    # dataset_name interpolates n_simulations; preprocessing steps interpolate adapter keys.
    assert cfg.data.dataset_name == f"training_data_{cfg.data.n_simulations}.npz"
    steps = OmegaConf.to_container(cfg.preprocessing.steps, resolve=True)
    names = [s["name"] for s in steps]
    assert names == ["drop_nan", "train_val_split", "standardize"]
    assert steps[0]["keys"] == ["x"]  # == adapter.summary_variables (derived from the simulator)


def test_adapter_derived_from_simulator(compose):
    # Empty adapter lists are filled from the simulator's declaration at CLI entry.
    raw = compose(fill=False)
    assert list(raw.adapter.inference_variables) == []
    assert list(raw.adapter.summary_variables) == []

    filled = compose()
    assert list(filled.adapter.inference_variables) == ["theta1", "theta2"]
    assert list(filled.adapter.summary_variables) == ["x"]


def test_adapter_explicit_config_wins(compose):
    cfg = compose(["adapter.inference_variables=[a,b,c]"])
    assert list(cfg.adapter.inference_variables) == ["a", "b", "c"]
    assert list(cfg.adapter.summary_variables) == ["x"]  # still derived where left empty


def test_group_override(compose):
    cfg = compose(["model/inference_network=diffusion"])
    assert cfg.model.inference_network.type == "diffusion"
    assert cfg.model.inference_network.time_embedding_dim == 32
