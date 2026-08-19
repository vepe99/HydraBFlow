"""Checks for the protoplanetary-disk arm: labels, adapter/network agreement, one training step.

No real data and no GPU: the RT caches are written synthetically into ``tmp_path`` and the batch is
4 rows on a small grid. What this pins is the wiring -- that the reader's shapes are what the
adapter expects, that the instrument model's output keys are what the adapter renames, that the
four image branches and their routed conditions line up, and that a gradient actually reaches the
weights. It does not pin any physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrabflow.simulators import _protoplan_spec as spec

pytest.importorskip("bayesflow")


N_ROWS = 12
N_SED = 19
GRID = 32                      # model image grid; small so the PSF/beam FFTs are cheap


# ══════════════════════════════════════════════════════════════════════════════════════
# Fixtures: a synthetic stand-in for the RT caches
# ══════════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def data_path(tmp_path):
    """A `split_combined/`-shaped directory of five `.npy` caches."""
    rng = np.random.default_rng(0)
    n_cols = 1 + len(spec.PARAMS)          # the leading "num" placeholder column
    params = rng.uniform(0.1, 1.0, size=(N_ROWS, n_cols))

    i_sil = 1 + spec.PARAMS.index("sil_id")
    i_cav = 1 + spec.PARAMS.index("log_r_cav")
    params[:, i_sil] = np.tile([0.0, 1.0], N_ROWS // 2)
    # Half the rows carry the no-cavity sentinel exactly, as the raw grid does.
    params[:, i_cav] = np.where(np.arange(N_ROWS) % 4 < 2, spec.NO_CAVITY_LOG_R_CAV, 1.0)

    np.save(tmp_path / "params_combined.npy", params)
    np.save(tmp_path / "sed_flx_jy_combined.npy",
            rng.uniform(1e-3, 1e-1, size=(N_ROWS, N_SED)))
    np.save(tmp_path / "sed_lams_combined.npy",
            np.tile(np.logspace(-0.3, 3.1, N_SED), (N_ROWS, 1)))
    np.save(tmp_path / "im_jy_combined_NHWC.npy",
            rng.uniform(0, 1e-4, size=(N_ROWS, GRID, GRID, 4)).astype(np.float32))
    np.save(tmp_path / "im_lams_combined.npy",
            np.tile([3.9, 450.0, 880.0, 1300.0], (N_ROWS, 1)))
    return tmp_path


@pytest.fixture
def raw_batch(data_path):
    from hydrabflow.simulators.protoplan import ProtoplanetaryDiskSimulator

    sim = ProtoplanetaryDiskSimulator(params={"data_path": str(data_path)})
    return sim, sim.load_dataset()


def _augmentor():
    """A cheap fixed-setup instrument model on the synthetic grid."""
    pytest.importorskip("stpsf")
    from hydrabflow.augmentation.protoplan_instrument import AugmentationsClass

    return AugmentationsClass(
        px_arcsec_mod=spec.FOV_ARCSEC_MOD / (GRID - 1),
        randomize_alma_setup=False,
        alma_beams=((0.20, 0.10, 0.0),) * 3,
        alma_noises_jy_beam=(8e-5, 8e-5, 8e-5),
        jwst_noise_mjy_sr=(0.5, 1.0),
        alma_obs_px_arcsec=0.05,
        seed=0,
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# The discrete encoding
# ══════════════════════════════════════════════════════════════════════════════════════

def test_labels_are_exact_and_shaped_by_role(data_path):
    """17 targets `(N,1)`, two conditions flat `(N,)` and exactly 0/1, no sentinel left over."""
    targets, columns, keep = spec.filter_and_jitter_params(
        data_path, rng=np.random.default_rng(0))

    assert targets == spec.NPE_TARGET_PARAMS and len(targets) == 17
    assert "sil_id" not in targets, "sil_id is a condition, not a target"
    for name in targets:
        assert columns[name].shape == (N_ROWS, 1), name

    # Flat (N,), not (N,1): the adapter expand_dims every scalar condition, so an (N,1) condition
    # would reach the concatenation as (N,1,1).
    for name in spec.DISCRETE_CONDITION_PARAMS:
        assert columns[name].shape == (N_ROWS,), name
        assert set(np.unique(columns[name])) <= {0.0, 1.0}, f"{name} was smeared"

    # has_cavity is derived from the *pre-redraw* log_r_cav, and log_r_cav no longer holds the
    # sentinel anywhere: on the no-cavity rows it is drawn from the real cavity branch, so
    # "posterior = prior there" is an honest statement rather than an off-support artefact.
    expected = np.where(np.arange(N_ROWS) % 4 < 2, 0.0, 1.0)
    np.testing.assert_array_equal(columns[spec.HAS_CAVITY], expected)
    log_r_cav = columns["log_r_cav"].reshape(-1)
    assert not np.any(log_r_cav == spec.NO_CAVITY_LOG_R_CAV)
    lo, hi = spec.CAVITY_LOG_R_CAV_RANGE
    assert np.all((log_r_cav[expected == 0] >= lo) & (log_r_cav[expected == 0] <= hi))
    assert keep.shape == (N_ROWS,) and keep.all()


def test_jitter_is_reproducible_from_its_seed(data_path):
    a = spec.filter_and_jitter_params(data_path, rng=np.random.default_rng(0))[1]["log_r_cav"]
    b = spec.filter_and_jitter_params(data_path, rng=np.random.default_rng(0))[1]["log_r_cav"]
    c = spec.filter_and_jitter_params(data_path, rng=np.random.default_rng(1))[1]["log_r_cav"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c), "a different seed must give different no-cavity draws"


def test_reader_shapes(raw_batch):
    _, data = raw_batch
    assert data["im_jy"].shape == (N_ROWS, GRID, GRID, 4)
    assert data["seds"].shape == (N_ROWS, N_SED, 1)
    assert data["sed_lams"].shape == (N_ROWS, N_SED, 1)
    assert data["im_lams"].shape == (N_ROWS, 4)


# ══════════════════════════════════════════════════════════════════════════════════════
# Layout: the adapter and the networks must agree, per band
# ══════════════════════════════════════════════════════════════════════════════════════

def test_condition_routing_is_per_band():
    """Each band's CNN sees its own beam/noise plus the shared geometry, and nothing else."""
    from hydrabflow.networks.protoplan import IMAGE_BRANCH_TAGS

    groups = spec.summary_condition_groups()
    routing = spec.summary_condition_routing()

    assert set(routing) == set(IMAGE_BRANCH_TAGS), "one routed branch per image key"
    assert "sed_input" not in routing, "the SED carries its per-bin noise as an input channel"

    for j, band in enumerate(spec.ALMA_BANDS):
        key = spec.ALMA_INPUT_KEYS[band]
        own = spec.ALMA_SUMMARY_COND_KEYS[band]
        assert routing[key] == [spec.GEOMETRY_SUMMARY_KEY, own]
        assert groups[own] == spec.alma_band_condition_keys(j)
        # No band may be told another band's beam.
        for other in spec.ALMA_BANDS:
            if other != band:
                assert not set(groups[own]) & set(
                    groups[spec.ALMA_SUMMARY_COND_KEYS[other]])

    # Every routed key is a member of summary_variables, so standardize="all" reaches it.
    assert set(groups) <= set(spec.summary_variable_keys())
    # The labels are never inputs.
    routed = {k for keys in groups.values() for k in keys}
    assert not routed & set(spec.DISCRETE_CONDITION_KEYS)


def test_adapter_and_summary_network_use_the_same_keys(compose):
    """The one real drift risk: the adapter's groups and the network's backbones are built by two
    different modules from two different config nodes."""
    from hydrabflow.networks.protoplan import IMAGE_BRANCH_TAGS, _summary_dims, _hp
    from hydrabflow.registry import build_summary_network

    cfg = compose(overrides=["experiment=protoplan"])
    net = build_summary_network(cfg.model.summary_network, cfg.adapter.summary_variables)

    assert set(net.backbones) == set(spec.summary_variable_keys()) - set(
        spec.summary_condition_groups())
    assert set(net.backbones) == set(IMAGE_BRANCH_TAGS) | {"sed_input"}
    assert net.routing == spec.summary_condition_routing()
    # `GroupedFlowMatching.group_sizes` is built from `_summary_dims`; its order must be the
    # concatenation order `ConditionedFusionNetwork.call` uses.
    assert sorted(_summary_dims(_hp(cfg.model.summary_network))) == sorted(net.backbones)


def test_grouped_flow_matching_group_sizes(compose):
    """Six groups -- the discrete pair, then one per band/modality -- with the pair protected."""
    from hydrabflow.registry import build_inference_network

    cfg = compose(overrides=[
        "experiment=protoplan",
        "model.summary_network.params.fuse_head=false",
        "model.inference_network.params.missing_modality_prob=0.5",
    ])
    net = build_inference_network(cfg.model.inference_network)
    hp = cfg.model.summary_network.params.hp
    assert net.group_sizes == [
        2,                              # sil_id, has_cavity
        hp.summary_dim_alma,            # b6_input   (sorted(backbones) order)
        hp.summary_dim_alma,            # b7_input
        hp.summary_dim_alma,            # b9_input
        hp.summary_dim_jwst,
        hp.summary_dim_sed,
    ]
    assert net.always_observed_groups == 1, "the discrete conditions are not a modality"


def test_missing_modality_mask_drops_whole_groups_but_never_the_discrete_pair():
    """With p=1 the mask is group-coherent, keep_one leaves one droppable group, and the two
    leading columns stay observed."""
    from hydrabflow.networks.protoplan import GroupedFlowMatching

    sizes = [2, 4, 4, 4, 3, 5]
    net = GroupedFlowMatching(
        group_sizes=sizes, missing_modality_prob=1.0, always_observed_groups=1)
    mask = np.asarray(net.observed_condition_mask(16))

    assert mask.shape == (16, sum(sizes))
    assert np.all(mask[:, :2] == 1), "sil_id/has_cavity must never be dropped"

    edges = np.cumsum([0] + sizes)
    for lo, hi in zip(edges[:-1], edges[1:]):
        group = mask[:, lo:hi]
        # This is the whole point of the class: upstream's per-element dropout would essentially
        # never take a modality's slice out together (prob = p ** width).
        assert np.all(group == group[:, :1]), "a group must be dropped or kept whole"
    per_group = np.stack([mask[:, lo] for lo, hi in zip(edges[:-1], edges[1:])], axis=-1)
    assert np.all(per_group[:, 1:].sum(axis=-1) == 1), "keep_one leaves exactly one at p=1"


# ══════════════════════════════════════════════════════════════════════════════════════
# End to end on synthetic rows
# ══════════════════════════════════════════════════════════════════════════════════════

def test_instrument_model_output_keys_match_the_adapter(raw_batch):
    """The augmentation's key names are the adapter's inputs; a rename drift is silent otherwise."""
    _, data = raw_batch
    batch = _augmentor()({k: np.asarray(v) for k, v in data.items()})

    for j in range(spec.N_ALMA):
        assert batch[f"im_jy_alma_{j}"].shape[-1] == 1, "one channel per band, own CNN"
    assert "im_jy" not in batch, "apply_resampling pops it"

    needed = (["im_jy_jwst", "sigma_sed_flux", "seds", "sed_lams"]
              + [f"im_jy_alma_{j}" for j in range(spec.N_ALMA)]
              + [k for keys in spec.summary_condition_groups().values() for k in keys])
    missing = [k for k in needed if k not in batch]
    assert not missing, f"the adapter reads keys the augmentation does not produce: {missing}"

    for key in needed:
        assert np.isfinite(np.asarray(batch[key])).all(), f"{key} is not finite"


def test_one_training_step(compose, raw_batch, tmp_path):
    """Adapter -> four CNNs + transformer -> flow, and a gradient that actually moves weights."""
    import keras

    from hydrabflow.pipeline.workflow import build_workflow

    cfg = compose(overrides=["experiment=protoplan"])
    _, data = raw_batch
    batch = _augmentor()({k: np.asarray(v) for k, v in data.items()})
    batch = {k: np.asarray(v) for k, v in batch.items()}

    workflow = build_workflow(cfg)

    # `standardize: [all]` must have survived the list -> string collapse. A one-element `["all"]`
    # reaches `Standardization` as a list whose only member matches no variable group, so it
    # standardizes *nothing* -- and nothing raises. Checked before `fit`, which expands the string
    # into the explicit group list.
    assert workflow.approximator.standardizer.standardize == "all"

    history = workflow.fit_offline(batch, epochs=1, batch_size=4, verbose=0)
    loss = history.history["loss"][-1]
    assert np.isfinite(loss), f"loss is {loss}"
    assert workflow.approximator.trainable_weights, "nothing to train"
    assert set(workflow.approximator.standardizer.standardize) == {
        "inference_variables", "summary_variables", "inference_conditions"}
    del keras


def test_one_training_step_with_missing_modality_dropout(compose, raw_batch):
    """The `group_sizes` width must equal the resolved condition width, or `ops.repeat` silently
    builds a mask of the wrong shape. Only a real forward pass catches that."""
    from hydrabflow.pipeline.workflow import build_workflow

    cfg = compose(overrides=[
        "experiment=protoplan",
        "model.summary_network.params.fuse_head=false",
        "model.inference_network.params.missing_modality_prob=0.4",
    ])
    _, data = raw_batch
    batch = {k: np.asarray(v) for k, v in _augmentor()(
        {k: np.asarray(v) for k, v in data.items()}).items()}

    workflow = build_workflow(cfg)
    history = workflow.fit_offline(batch, epochs=1, batch_size=4, verbose=0)
    assert np.isfinite(history.history["loss"][-1])


def test_missing_modality_dropout_rejects_a_fused_head(compose, raw_batch):
    """An MLP head mixes the branch embeddings, so there is no modality boundary left to drop --
    and the widths no longer line up either. That must raise, not train something else."""
    from hydrabflow.pipeline.workflow import build_workflow

    cfg = compose(overrides=[
        "experiment=protoplan",
        "model.summary_network.params.fuse_head=true",          # the mistake
        "model.inference_network.params.missing_modality_prob=0.4",
        "model.summary_network.params.hp.FusionNetwork_final_summary_dimension=8",
    ])
    _, data = raw_batch
    batch = {k: np.asarray(v) for k, v in _augmentor()(
        {k: np.asarray(v) for k, v in data.items()}).items()}

    with pytest.raises(ValueError, match="group_sizes sums to"):
        build_workflow(cfg).fit_offline(batch, epochs=1, batch_size=4, verbose=0)


def test_saved_approximator_reloads_with_its_weights(compose, raw_batch, tmp_path):
    """`evaluate` must get the *trained* weights back, not a reinitialized architecture.

    The plain `.keras` load cannot do that for this model -- see `artifacts.load_approximator`:
    any dict-of-backbones fusion network (stock `bf.networks.FusionNetwork` included) saves its
    branches under `layers/...` and reloads looking under `summary_network/...`, so every branch
    comes back empty. This pins the rebuild-from-config path `evaluate` relies on instead; without
    it the failure is silent, and only a garbage posterior would show it.
    """
    from hydrabflow.pipeline import artifacts
    from hydrabflow.pipeline.workflow import BEST_WEIGHTS_NAME, build_workflow

    cfg = compose(overrides=["experiment=protoplan"])
    _, data = raw_batch
    batch = _augmentor()({k: np.asarray(v) for k, v in data.items()})
    batch = {k: np.asarray(v) for k, v in batch.items()}

    workflow = build_workflow(cfg)
    workflow.fit_offline(batch, epochs=1, batch_size=4, verbose=0)
    artifacts.save_approximator(workflow, str(tmp_path))
    workflow.approximator.save_weights(
        str(tmp_path / f"{BEST_WEIGHTS_NAME}.weights.h5"))    # what the fit callback writes
    trained = [np.asarray(w) for w in workflow.approximator.weights]
    assert trained, "nothing was trained"

    reloaded = artifacts.load_approximator(
        str(tmp_path), workflow=build_workflow(cfg),
        probe_batch={k: v[:2] for k, v in batch.items()})
    restored = [np.asarray(w) for w in reloaded.weights]

    # Compared in tree order: `w.path` embeds autogenerated layer names, which differ per instance.
    assert len(trained) == len(restored)
    for i, (a, b) in enumerate(zip(trained, restored)):
        np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-6,
                                   err_msg=f"weight {i} did not survive save + load")


def test_pixel_scale_mismatch_raises(raw_batch):
    """The one config error that would otherwise be silent: a differently-sampled image cache with
    the old `px_arcsec_mod`, i.e. the right units convolved with the wrong beam."""
    from hydrabflow.augmentation.protoplan_instrument import AugmentationsClass

    pytest.importorskip("stpsf")
    _, data = raw_batch
    wrong = AugmentationsClass(
        px_arcsec_mod=spec.FOV_ARCSEC_MOD / (128 - 1),   # a 128-pixel cache's spacing
        randomize_alma_setup=False, alma_obs_px_arcsec=0.05, seed=0,
    )
    with pytest.raises(ValueError, match="silent beam error"):
        wrong({k: np.asarray(v) for k, v in data.items()})   # but the batch is 32x32


def test_missing_modality_mask_actually_changes_the_output():
    """The mask must reach the network, under either subnet.

    Regression test for a silent no-op: `FlowMatching.__init__` keeps only the mask kwargs its
    subnet's `call` accepts, and every MLP-style subnet (including the `time_mlp` default) accepts
    none. `missing_modality_prob` therefore built a mask every step and changed nothing, and a
    masked evaluation matched an unmasked one to six decimals. Asserting the *loss moves* is the
    check that would have caught it; asserting the mask is merely generated would not have.
    """
    import keras
    import numpy as np
    from bayesflow.utils import MaskName
    from hydrabflow.networks.protoplan import GroupedFlowMatching

    sizes, names = [2, 8, 8], ["discrete", "a_input", "b_input"]
    rng = np.random.default_rng(0)
    x = keras.ops.convert_to_tensor(rng.normal(size=(16, 3)).astype("float32"))
    cond = keras.ops.convert_to_tensor(rng.normal(size=(16, sum(sizes))).astype("float32"))
    mask = np.ones((16, sum(sizes)), dtype="float32")
    mask[:, 2:10] = 0.0                       # drop "a_input"
    mask = keras.ops.convert_to_tensor(mask)

    for subnet, kwargs in [("time_mlp", {"widths": [16, 16]}),
                           ("diffusion_transformer", {"widths": [16, 16], "num_heads": 2})]:
        net = GroupedFlowMatching(subnet=subnet, subnet_kwargs=kwargs, group_sizes=sizes,
                                  group_names=names, missing_modality_prob=0.0,
                                  always_observed_groups=1)
        net.build(keras.ops.shape(x), keras.ops.shape(cond))
        free = net.compute_metrics(x, conditions=cond, stage="validation")["loss"]
        held = net.compute_metrics(x, conditions=cond, stage="validation",
                                   **{MaskName.OBSERVED_CONDITION: mask})["loss"]
        assert not np.isclose(float(free), float(held), rtol=1e-4), (
            f"{subnet}: masking a group left the loss unchanged "
            f"({float(free):.6f} vs {float(held):.6f}) -- the mask is being discarded")


def test_av_accepts_a_scalar_or_a_per_disk_range(raw_batch):
    """`av: [lo, hi]` draws one extinction per disk; `av: x` is the old batch-wide constant."""
    from hydrabflow.augmentation.protoplan_instrument import AugmentationsClass

    _, data = raw_batch
    batch = {k: np.asarray(v) for k, v in data.items()}

    def preprocessed(av):
        aug = AugmentationsClass(
            px_arcsec_mod=spec.FOV_ARCSEC_MOD / (GRID - 1), av=av, has_jwst=False,
            randomize_alma_setup=False, alma_obs_px_arcsec=0.05, seed=0,
        )
        return np.asarray(aug.preprocess(batch)["seds"])

    # A degenerate range reproduces the scalar exactly: same law, same factorisation.
    np.testing.assert_allclose(preprocessed(4.0), preprocessed([4.0, 4.0]), rtol=1e-5)

    # A real range makes the per-disk attenuation differ, and more A_V means less flux.
    scalar, sampled = preprocessed(0.0), preprocessed([1.0, 10.0])
    ratio = (sampled / scalar)[:, 0, 0]
    assert ratio.std() > 0                                # not one constant for the whole batch
    assert np.all(ratio > 0) and np.all(ratio < 1)        # extinction only ever dims
