"""The two conditioned summary networks and the modality-dropout flow, plus their builders.

Ported from `ProtoplanetaryDisk_SBI`'s `npe/networks.py` and `npe/setup.build_networks`.

Why these exist
---------------
Every non-physical scalar the instrument augmentation produces -- the in-plane rotation, the JWST
noise sigma, the three ALMA beam FWHMs/PAs and their noise sigmas -- used to be concatenated into
`inference_conditions`, i.e. handed to the flow *after* the summary network had already compressed
the images to a ~16-64 dimensional vector.  That ordering asks the wrong network to do the work:
whether a faint ring is real or a noise fluctuation, and how much a beam has smeared it, is a
question about the *image*, and the only network that ever sees the image is the CNN.  By the time
the noise level reaches the inference network the image has been summarised without it.

So those scalars go into the summary networks instead, split per instrument by
`_protoplan_spec.summary_condition_routing`.  `inference_conditions` keeps only the genuinely
discrete physical parameters (`sil_id`, `has_cavity`), which are conditions on the *density*
rather than descriptions of the measurement.

Where the concatenation happens
-------------------------------
At the end of the convolutional trunk -- after the conv stages and the pooling head but *before*
the projection to `summary_dim`.  That is the widest, least-compressed vector in the branch, so
the noise scalars get to interact with image features while there are still image features to
interact with; concatenating onto the already-projected `summary_dim` vector would be a much
narrower bottleneck, and concatenating onto the image tensor itself would spend conv capacity on
spatially-constant channels.

One coupling to upstream, guarded
---------------------------------
`ConvolutionalNetwork` exposes its stack as a flat `self.layers` list ending in
`Dense(summary_dim)`.  `ConditionedConvolutionalNetwork` reuses everything but that final `Dense`
and checks in `__init__` that the last layer really is one -- so an upstream change to the head
raises here instead of silently dropping a pooling layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import bayesflow as bf
import keras
import numpy as np
from bayesflow.networks.summary.summary_network import SummaryNetwork
from bayesflow.utils import MaskName
from bayesflow.utils.masks import random_mask
from bayesflow.utils.serialization import deserialize, serializable, serialize
from keras import ops

from hydrabflow.registry import register_inference_network, register_summary_network
from hydrabflow.simulators._protoplan_spec import (
    ALMA_BANDS,
    ALMA_INPUT_KEYS,
    DISCRETE_CONDITION_KEYS,
    summary_condition_routing,
)


#: Name of the leading, never-dropped condition group (`sil_id` + `has_cavity`). The remaining
#: group names are the summary backbone keys, so `eval.mask_condition_groups: [b7_input]` reads
#: the same as the adapter's key for that band.
DISCRETE_GROUP_NAME = "discrete"


@serializable("hydrabflow.networks.protoplan")
class ConditionedConvolutionalNetwork(SummaryNetwork):
    """
    A `ConvolutionalNetwork` whose pooled features are concatenated with a condition vector
    and then passed through an MLP before the projection to `summary_dim`.

    Call signature is `layer(image, conditions=...)`.  `conditions` is `(B, D)`; `None` makes
    the layer behave like the stock network with an extra MLP in front of the projection,
    which is what an unconditioned ablation gets.

    Parameters
    ----------
    summary_dim : int
        Output width, as in `ConvolutionalNetwork`.
    mlp_widths : sequence of int
        The post-concatenation MLP.  Its job is to let the noise scalars *modulate* the image
        features rather than be summed into them, which a single linear projection cannot do.
    dropout : float
        Shared by the convolutional trunk and the MLP.
    cnn_kwargs : dict
        Everything else forwarded to `ConvolutionalNetwork` (`widths`, `blocks_per_stage`,
        `down_mode`, `pool_head`, ...).
    """

    def __init__(self, summary_dim: int = 16, mlp_widths: Sequence[int] = (64,),
                 dropout: float = 0.05, cnn_kwargs: Mapping | None = None, **kwargs):
        super().__init__(**kwargs)
        self.summary_dim = int(summary_dim)
        self.mlp_widths = list(mlp_widths)
        self.dropout = float(dropout)
        self.cnn_kwargs = dict(cnn_kwargs or {})

        cnn = bf.networks.ConvolutionalNetwork(
            summary_dim=self.summary_dim, dropout=self.dropout, **self.cnn_kwargs
        )
        if not isinstance(cnn.layers[-1], keras.layers.Dense):
            raise TypeError(
                "ConvolutionalNetwork's layer stack no longer ends in the "
                f"Dense(summary_dim) projection (found {type(cnn.layers[-1])!r}). "
                "ConditionedConvolutionalNetwork drops that last layer to splice the "
                "conditions in before it; update it to match upstream's head."
            )
        # Keep the trunk *layers*, not the `ConvolutionalNetwork` that made them.  Holding on
        # to the wrapper would leave two tracked-but-never-called layers behind — itself and
        # the dropped Dense — and keras's `_symbolic_build` triggers on *any* unbuilt sublayer,
        # then tries to rebuild the whole approximator by calling it on the raw batch dict.
        # The failure surfaces four frames down inside `Standardize`, so it is worth avoiding.
        self._trunk = list(cnn.layers[:-1])

        self.mlp = bf.networks.MLP(widths=self.mlp_widths, dropout=self.dropout)
        self.projection = keras.layers.Dense(self.summary_dim)

    def build(self, image_shape, conditions_shape=None):
        if self.built:
            return
        x = ops.zeros((1, *tuple(image_shape)[1:]))
        c = None if conditions_shape is None else ops.zeros((1, *tuple(conditions_shape)[1:]))
        self.call(x, conditions=c)
        self.built = True

    def compute_output_shape(self, image_shape, conditions_shape=None):
        return (tuple(image_shape)[0], self.summary_dim)

    # `image`, not `x`: keras derives `build`'s kwargs from `call`'s parameter names, so an
    # `image_shape` argument requires an `image` one here.
    def call(self, image, conditions=None, training: bool = False):
        x = image
        for layer in self._trunk:
            x = layer(x, training=training)
        if conditions is not None:
            x = ops.concatenate([x, ops.cast(conditions, x.dtype)], axis=-1)
        x = self.mlp(x, training=training)
        return self.projection(x)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "summary_dim": self.summary_dim,
            "mlp_widths": self.mlp_widths,
            "dropout": self.dropout,
            "cnn_kwargs": self.cnn_kwargs,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@serializable("hydrabflow.networks.protoplan")
class GroupedFlowMatching(bf.networks.FlowMatching):
    """
    `FlowMatching` with modality-coherent condition dropout, in place of stock
    `missing_conditions_prob`.

    Upstream's `sample_input_masks` drops each element of the resolved condition vector
    independently, so an entire modality's slice of the fused summary embedding is
    essentially never dropped together (probability `drop_prob ** summary_dim`) -- that
    trains resilience to occasional feature dropout, not to a genuinely missing modality.
    `sample_input_masks` only auto-generates `observed_condition_mask` when the subnet kwarg
    isn't already supplied, so this class just supplies its own group-coherent draw before
    delegating to the stock implementation. No change to `DiffusionTransformer` (it only
    consumes the mask) or to `bayesflow.utils.masks` is needed.

    Parameters
    ----------
    group_sizes : sequence of int
        Widths of each condition group, in the order they appear in the resolved condition
        vector, i.e. `inference_conditions` followed by the summary network's output
        (`ConditionBuilder.resolve`). Must sum to the condition width.

        For this to mean anything the summary network's output has to *have* group boundaries,
        which rules out a `FusionNetwork` head: an MLP over the concatenated branch embeddings
        mixes them, and dropping a slice of the mixed vector is not dropping a modality. So
        `build_summary_network` sets `head=None` whenever this is enabled, and the summary output
        is the raw concatenation of the branch embeddings in `sorted(backbones)` order.
    missing_modality_prob : float
        Per-group, per-sample drop probability during training. `0.0` (default) disables
        this and the class behaves exactly like `FlowMatching`.
    always_observed_groups : int
        Leading groups that are never dropped. `inference_conditions` here holds `sil_id` and
        `has_cavity`, which are conditions on the *density*, not modalities of the measurement:
        dropping them asks the network to marginalise over a parameter it is meant to be
        conditioned on. `keep_one=True` alone does not protect them -- it only guarantees that
        *some* group survives.
    """

    def __init__(self, *args, group_sizes: Sequence[int], missing_modality_prob: float = 0.0,
                 always_observed_groups: int = 0, group_names: Sequence[str] = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.group_sizes = [int(s) for s in group_sizes]
        self.missing_modality_prob = float(missing_modality_prob)
        self.always_observed_groups = int(always_observed_groups)
        # Names in the same order as `group_sizes`, so an *evaluation* can address one modality by
        # name (`eval.mask_condition_groups`) instead of by column offset. Empty is allowed and
        # only costs that knob -- an older checkpoint deserializes without it.
        self.group_names = [str(n) for n in group_names]
        if self.group_names and len(self.group_names) != len(self.group_sizes):
            raise ValueError(
                f"group_names has {len(self.group_names)} entries but group_sizes has "
                f"{len(self.group_sizes)}; they index the same groups.")

    def observed_condition_mask(self, batch_size: int):
        """`(batch_size, sum(group_sizes))`, 0 on every column of a dropped group.

        Split out of `compute_metrics` so it is testable without building the network.

        The draw covers the *droppable* groups only, and the always-observed ones are prepended
        afterwards. Drawing over all of them and then forcing the leading ones back to 1 would let
        `keep_one` spend its guarantee on a group that was never at risk, leaving rows with every
        modality dropped -- pure-prior training samples, at a rate of roughly
        `always_observed / n_groups` when `missing_modality_prob` is high.
        """
        n = self.always_observed_groups
        droppable = self.group_sizes[n:]
        if not droppable:
            return ops.ones((batch_size, sum(self.group_sizes)))

        keep = random_mask(
            (batch_size, len(droppable)), self.missing_modality_prob,
            keep_one=True, seed_generator=self.seed_generator,
        )
        if n:
            keep = ops.concatenate([ops.ones_like(ops.repeat(keep[:, :1], n, axis=-1)), keep],
                                   axis=-1)
        return ops.repeat(keep, np.array(self.group_sizes), axis=-1)

    def _apply(self, conditions, mask):
        """Zero the columns of `conditions` that `mask` marks unobserved.

        Upstream applies `observed_condition_mask` *inside the subnet*, and `FlowMatching.__init__`
        keeps only the mask kwargs the subnet's `call` actually accepts
        (`_subnet_mask_keys`) -- which for every MLP-style subnet, including the `time_mlp`
        default used here, is the empty set. The mask is then silently dropped, so
        `missing_modality_prob` would build a mask each step and change nothing. Only
        `DiffusionTransformer` consumes it, because it masks attention over condition *tokens*.

        A flat MLP has no tokens to mask, so "missing" is encoded the only way it can be: the
        modality's slice of the condition vector is zeroed, and the network learns that an
        all-zero slice means absent. Applied here, before delegating, so it works whatever the
        subnet is.

        Skipped entirely when the subnet *does* accept the mask (`diffusion_transformer`): there
        each scalar of the condition vector is its own token, and the mask excludes those tokens
        from attention -- a strictly better "absent" than a zeroed value, which the model would
        still attend to. Zeroing on top of that would destroy the value the token carries for no
        gain, so the two mechanisms are exclusive rather than cumulative.
        """
        if conditions is None or mask is None:
            return conditions
        if MaskName.OBSERVED_CONDITION in getattr(self, "_subnet_mask_keys", ()):
            return conditions
        return conditions * ops.cast(mask, ops.dtype(conditions))

    def compute_metrics(self, x, conditions=None, sample_weight=None, stage: str = "training", **kwargs):
        if stage == "training" and self.missing_modality_prob > 0 and MaskName.OBSERVED_CONDITION not in kwargs:
            width = int(ops.shape(conditions)[-1])
            if width != sum(self.group_sizes):
                # Silent otherwise: a too-narrow mask broadcasts, so the wrong columns get zeroed
                # and training just quietly optimises a different objective.
                raise ValueError(
                    f"group_sizes sums to {sum(self.group_sizes)} but the resolved condition "
                    f"vector is {width} wide. It must be inference_conditions followed by the "
                    "summary network's output, one entry per group -- and the summary network "
                    "must expose per-modality boundaries (head=None / fuse_head: false)."
                )
            kwargs[MaskName.OBSERVED_CONDITION] = self.observed_condition_mask(
                ops.shape(conditions)[0])
        conditions = self._apply(conditions, kwargs.get(MaskName.OBSERVED_CONDITION))
        return super().compute_metrics(x, conditions=conditions, sample_weight=sample_weight, stage=stage, **kwargs)

    def velocity(self, xz, time, conditions=None, training: bool = False, **kwargs):
        """The sampling-time counterpart of the masking in `compute_metrics`.

        `_inverse` hands its `**kwargs` to this on every integrator step, so an
        `observed_condition_mask` passed to `sample()` (as `eval.mask_condition_groups` does)
        reaches here already sliced per batch and repeated per posterior draw.
        """
        conditions = self._apply(conditions, kwargs.get(MaskName.OBSERVED_CONDITION))
        return super().velocity(xz, time, conditions=conditions, training=training, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "group_sizes": self.group_sizes,
            "missing_modality_prob": self.missing_modality_prob,
            "always_observed_groups": self.always_observed_groups,
            "group_names": self.group_names,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@serializable("hydrabflow.networks.protoplan")
class ConditionedFusionNetwork(SummaryNetwork):
    """
    `FusionNetwork` plus a routing table saying which entries of `summary_variables` are
    *conditions* for which backbone.

    `backbones` maps an input key to its network, exactly as upstream.  `routing` maps a
    backbone key to the ordered list of condition keys concatenated (last axis) and passed to
    that backbone as `conditions=`.  Keys of `summary_variables` that appear only in `routing`
    have no backbone of their own and are never summarised on their own -- which is the point:
    a beam FWHM is not a modality, it is a description of one.

    A backbone absent from `routing` is called as a plain summary network, so the SED
    `TimeSeriesTransformer` needs no wrapper (its per-bin noise already arrives as an input
    channel of `sed_input`).

    Late fusion is unchanged: branch outputs are concatenated and passed through `head`.
    """

    def __init__(self, backbones: Mapping[str, keras.Layer],
                 routing: Mapping[str, Sequence[str]] | None = None,
                 head: keras.Layer | None = None, **kwargs):
        super().__init__(**kwargs)
        self.backbones = dict(backbones)
        self.routing = {k: list(v) for k, v in (routing or {}).items()}
        self.head = head
        self._ordered_keys = sorted(self.backbones.keys())

        unknown = set(self.routing) - set(self.backbones)
        if unknown:
            raise ValueError(
                f"routing refers to keys with no backbone: {sorted(unknown)}")

    # ── shapes ───────────────────────────────────────────────────────────────
    def _conditions_shape(self, key: str, inputs_shape: Mapping[str, tuple]):
        """`(B, sum_of_widths)` for a routed backbone, `None` for an unrouted one."""
        cond_keys = self.routing.get(key)
        if not cond_keys:
            return None
        shapes = [inputs_shape[c] for c in cond_keys]
        return (shapes[0][0], sum(int(s[-1]) for s in shapes))

    def _conditions(self, key: str, inputs: Mapping[str, "keras.KerasTensor"]):
        cond_keys = self.routing.get(key)
        if not cond_keys:
            return None
        if len(cond_keys) == 1:
            return inputs[cond_keys[0]]
        return ops.concatenate([inputs[c] for c in cond_keys], axis=-1)

    def build(self, inputs_shape: Mapping[str, tuple]):
        if self.built:
            return
        output_shapes = []
        for key in self._ordered_keys:
            backbone = self.backbones[key]
            cond_shape = self._conditions_shape(key, inputs_shape)
            if not backbone.built:
                if cond_shape is None:
                    backbone.build(inputs_shape[key])
                else:
                    backbone.build(inputs_shape[key], cond_shape)
            if cond_shape is None:
                output_shapes.append(backbone.compute_output_shape(inputs_shape[key]))
            else:
                output_shapes.append(
                    backbone.compute_output_shape(inputs_shape[key], cond_shape))
        if self.head is not None and not self.head.built:
            fusion_shape = (*output_shapes[0][:-1],
                            sum(int(s[-1]) for s in output_shapes))
            self.head.build(fusion_shape)
        self.built = True

    def compute_output_shape(self, inputs_shape: Mapping[str, tuple]):
        output_shapes = []
        for key in self._ordered_keys:
            cond_shape = self._conditions_shape(key, inputs_shape)
            if cond_shape is None:
                output_shapes.append(
                    self.backbones[key].compute_output_shape(inputs_shape[key]))
            else:
                output_shapes.append(
                    self.backbones[key].compute_output_shape(inputs_shape[key], cond_shape))
        output_shape = (*output_shapes[0][:-1], sum(int(s[-1]) for s in output_shapes))
        if self.head is not None:
            output_shape = self.head.compute_output_shape(output_shape)
        return output_shape

    # ── forward ──────────────────────────────────────────────────────────────
    def _branch_outputs(self, inputs, training: bool):
        outputs = []
        for key in self._ordered_keys:
            cond = self._conditions(key, inputs)
            if cond is None:
                outputs.append(self.backbones[key](inputs[key], training=training))
            else:
                outputs.append(
                    self.backbones[key](inputs[key], conditions=cond, training=training))
        return outputs

    def call(self, inputs: Mapping[str, "keras.KerasTensor"], training: bool = False):
        outputs = ops.concatenate(self._branch_outputs(inputs, training), axis=-1)
        if self.head is None:
            return outputs
        return self.head(outputs, training=training)

    def compute_metrics(self, inputs: Mapping[str, "keras.KerasTensor"],
                        stage: str = "training", **kwargs) -> dict:
        """
        Mirrors `FusionNetwork.compute_metrics`, including collecting an optional
        `base_distribution` MMD loss from any backbone that is itself a `SummaryNetwork`.
        Conditioned backbones are plain layers and contribute no loss.
        """
        if not self.built:
            self.build(keras.tree.map_structure(ops.shape, inputs))

        training = stage == "training"
        metrics = {"loss": [], "outputs": []}
        for key in self._ordered_keys:
            backbone = self.backbones[key]
            cond = self._conditions(key, inputs)
            if cond is None and isinstance(backbone, SummaryNetwork):
                metrics_k = backbone.compute_metrics(inputs[key], stage=stage, **kwargs)
                metrics["outputs"].append(metrics_k["outputs"])
                if "loss" in metrics_k:
                    metrics["loss"].append(metrics_k["loss"])
            elif cond is None:
                metrics["outputs"].append(backbone(inputs[key], training=training))
            else:
                metrics["outputs"].append(
                    backbone(inputs[key], conditions=cond, training=training))

        if len(metrics["loss"]) == 0:
            del metrics["loss"]
        else:
            metrics["loss"] = ops.sum(metrics["loss"])

        metrics["outputs"] = ops.concatenate(metrics["outputs"], axis=-1)
        if self.head is not None:
            metrics["outputs"] = self.head(metrics["outputs"], training=training)
        return metrics

    def get_config(self):
        base_config = super().get_config()
        config = {
            "backbones": self.backbones,
            "routing": self.routing,
            "head": self.head,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


# ══════════════════════════════════════════════════════════════════════════════════════
# Builders
# ══════════════════════════════════════════════════════════════════════════════════════

#: A small architecture: one convolutional stage for JWST, two for ALMA, narrow CNN /
#: transformer / MLP widths throughout.  The default because it is what the disk runs so far have
#: used; `pipeline/tune.py` can address any of these keys as
#: `model.summary_network.params.hp.<key>` in its `search_space`.
SMALL_HYPERPARAMETERS = dict(
    summary_dim_jwst=16, cnn_num_stages_jwst=1, cnn_base_width_jwst=8,
    cnn_blocks_per_stage_jwst=1, cnn_down_mode_jwst="avg_pool",
    cnn_cond_mlp_width_jwst=32, cnn_cond_mlp_depth_jwst=1,
    # Shared by all three ALMA bands unless a `*_b9`/`*_b7`/`*_b6` key overrides it -- see `_tag`.
    summary_dim_alma=16, cnn_num_stages_alma=2, cnn_base_width_alma=8,
    cnn_blocks_per_stage_alma=1, cnn_down_mode_alma="avg_pool",
    cnn_cond_mlp_width_alma=32, cnn_cond_mlp_depth_alma=1,
    summary_dim_sed=16,
    tst_num_layers=1, tst_num_heads=1, tst_head_dim=16,
    tst_expansion_factor=2.0, tst_time_embed_dim=8,
    FusionNetwork_width_head=32, FusionNetwork_length_head=1,
    FusionNetwork_final_summary_dimension=16,
    inference_mlp_depth=2, inference_mlp_width=32,
    inference_time_embedding_dim=16,
)


def _hp(cfg) -> dict:
    """The architecture dict out of `cfg.params.hp`, defaulted key by key.

    Defaulted rather than required so a partial override in config -- or a tuning trial that
    searches three keys -- still builds.
    """
    from omegaconf import OmegaConf

    given = cfg.params.get("hp", {})
    if OmegaConf.is_config(given):
        given = OmegaConf.to_container(given, resolve=True)
    return {**SMALL_HYPERPARAMETERS, **dict(given or {})}


def _tag(hp: dict, name: str, tag: str):
    """`hp[f"{name}_{tag}"]`, falling back to the shared `_alma` value for an ALMA band.

    So three bands cost three keys in config only where they should actually differ (a band with a
    much finer beam wanting a deeper trunk, say); by default one `*_alma` value describes all
    three, and `tuning.search_space` can address either level.
    """
    key = f"{name}_{tag}"
    if key not in hp and tag in ALMA_BANDS:
        key = f"{name}_alma"
    return hp[key]


#: `{input key: hp tag}` for the four image branches (JWST + the three ALMA bands).
IMAGE_BRANCH_TAGS = {"jwst_input": "jwst",
                     **{ALMA_INPUT_KEYS[b]: b for b in ALMA_BANDS}}


def _summary_dims(hp: dict) -> dict[str, int]:
    """Per-backbone output width, keyed by input key.  Also the `group_sizes` source."""
    dims = {key: int(_tag(hp, "summary_dim", tag)) for key, tag in IMAGE_BRANCH_TAGS.items()}
    dims["sed_input"] = int(hp["summary_dim_sed"])
    return dims


@register_summary_network("protoplan_fusion")
def _protoplan_fusion(cfg):
    """One conditioned CNN per image band (JWST + ALMA B9/B7/B6) + a SED transformer, late-fused.

    `params.fuse_head` (default True) selects the MLP head.  False returns the raw concatenation
    of the branch embeddings, which is what `GroupedFlowMatching` needs to be able to drop a whole
    modality -- set both or neither.
    """
    hp = _hp(cfg)
    dropout = float(cfg.dropout)
    dims = _summary_dims(hp)

    backbones = {}
    for key, tag in IMAGE_BRANCH_TAGS.items():
        widths = tuple(
            int(_tag(hp, "cnn_base_width", tag)) * (2 ** i)
            for i in range(int(_tag(hp, "cnn_num_stages", tag)))
        )
        depth = int(_tag(hp, "cnn_cond_mlp_depth", tag))
        backbones[key] = ConditionedConvolutionalNetwork(
            summary_dim=dims[key],
            mlp_widths=[int(_tag(hp, "cnn_cond_mlp_width", tag))] * depth,
            dropout=dropout,
            cnn_kwargs=dict(
                widths=widths,
                blocks_per_stage=int(_tag(hp, "cnn_blocks_per_stage", tag)),
                down_mode=_tag(hp, "cnn_down_mode", tag),
                pool_head="attention",
            ),
        )

    # The SED transformer is deliberately unrouted: its per-bin noise already arrives as an
    # input channel of `sed_input`, so it needs no condition vector and no wrapper.
    n_heads = int(hp["tst_num_heads"])
    n_layers = int(hp["tst_num_layers"])
    backbones["sed_input"] = bf.networks.TimeSeriesTransformer(
        summary_dim=dims["sed_input"],
        embed_dims=(n_heads * int(hp["tst_head_dim"]),) * n_layers,   # divisibility guaranteed
        num_heads=(n_heads,) * n_layers,
        dropout=dropout,
        expansion_factor=float(hp["tst_expansion_factor"]),
        time_embed_dim=int(hp["tst_time_embed_dim"]),
        time_axis=-1,
    )

    head = None
    if bool(cfg.params.get("fuse_head", True)):
        head = keras.Sequential([
            bf.networks.MLP(
                widths=[int(hp["FusionNetwork_width_head"])] * int(hp["FusionNetwork_length_head"])
            ),
            keras.layers.Dense(units=int(hp["FusionNetwork_final_summary_dimension"])),
        ])
    return ConditionedFusionNetwork(
        backbones=backbones, routing=summary_condition_routing(), head=head)


@register_inference_network("protoplan_flow_matching")
def _protoplan_flow_matching(cfg):
    """`FlowMatching`, or `GroupedFlowMatching` when `params.missing_modality_prob > 0`."""
    hp = _hp(cfg)
    # `time_mlp` (the upstream default) or `diffusion_transformer`. The choice is load-bearing for
    # `missing_modality_prob`: only the transformer consumes `observed_condition_mask` natively,
    # tokenizing each condition scalar and masking those tokens out of attention. With an MLP
    # subnet `GroupedFlowMatching` falls back to zeroing the group's columns.
    subnet = str(cfg.params.get("subnet", "time_mlp"))
    if subnet == "diffusion_transformer":
        # Only the keys the run actually sets; the rest fall through to
        # `DIFFUSION_TRANSFORMER_DEFAULTS` (5 x 128, 4 heads, expansion 3).
        subnet_kwargs = {}
        if "dt_width" in hp and "dt_num_layers" in hp:
            subnet_kwargs["widths"] = [int(hp["dt_width"])] * int(hp["dt_num_layers"])
        for key, cast in (("dt_num_heads", int), ("dt_expansion_factor", float),
                          ("dt_time_embedding_dim", int), ("dt_dropout", float)):
            if key in hp:
                subnet_kwargs[key[3:] if key != "dt_time_embedding_dim" else "time_embedding_dim"] \
                    = cast(hp[key])
    else:
        subnet_kwargs = {
            "widths": [int(hp["inference_mlp_width"])] * int(hp["inference_mlp_depth"]),
            "time_embedding_dim": int(hp["inference_time_embedding_dim"]),
        }
    prob = float(cfg.params.get("missing_modality_prob", 0.0))
    if prob <= 0.0:
        return bf.networks.FlowMatching(subnet=subnet, subnet_kwargs=subnet_kwargs)

    # `discrete_condition_groups: [sil_id, has_cavity]` (names in `adapter.inference_conditions`
    # order) makes each indicator its own width-1, *droppable* group, so a mask can marginalise
    # over one of them independently. Without it the indicators stay one always-observed group of
    # `discrete_condition_dim` -- the original behaviour. A count mismatch against the adapter is
    # caught by the width check in `compute_metrics` at the first step.
    per_indicator = list(cfg.params.get("discrete_condition_groups", []) or [])
    if per_indicator:
        discrete_sizes, discrete_names = [1] * len(per_indicator), [str(n) for n in per_indicator]
    else:
        discrete_sizes = [int(cfg.params.get("discrete_condition_dim",
                                             len(DISCRETE_CONDITION_KEYS)))]
        discrete_names = [DISCRETE_GROUP_NAME]
    dims = _summary_dims(hp)
    return GroupedFlowMatching(
        subnet=subnet,
        subnet_kwargs=subnet_kwargs,
        # `inference_conditions` first, then the summary output -- and with `fuse_head: false`
        # that output is the branch embeddings concatenated in `sorted(backbones)` order, which
        # `_summary_dims` shares its keys with. Six groups: the discrete pair, then B6, B7, B9,
        # JWST, SED.
        # Width of the always-observed leading group = how many discrete indicators the adapter
        # actually concatenates. Must match `adapter.inference_conditions`; a mismatch raises in
        # `compute_metrics` at the first step rather than silently masking the wrong columns.
        group_sizes=discrete_sizes + [dims[k] for k in sorted(dims)],
        group_names=discrete_names + sorted(dims),
        missing_modality_prob=prob,
        # 0 when the indicators are their own groups: that is the only reason to split them.
        always_observed_groups=0 if per_indicator else 1,
    )


#: This builder returns its own `FusionNetwork`, so `registry.build_summary_network` must not wrap
#: it in the generic one-backbone-per-summary-key fusion (the disk's summary keys are not one
#: modality each: three are modalities and three are routed condition groups).
_protoplan_fusion.handles_fusion = True
