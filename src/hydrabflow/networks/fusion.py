"""Multi-observable fusion summary network (attention-mask aware).

Consumes the dict the adapter builds when several observable keys are grouped into
``summary_variables``: one summary backbone per key, concatenated and optionally passed through
a small head MLP. Ports the reference project's ``utils/custom_summary_network.FusionNetwork``;
unlike ``bayesflow.networks.FusionNetwork`` it forwards the ``attention_mask`` the approximator
routes from the ``summary_attention_mask`` data key — but only to the designated set-valued
backbone (``mask_backbone``), since the other inputs (e.g. a rotation curve) have their own,
un-masked geometry.

Selected via ``model/summary_network``::

    type: fusion
    params:
      mask_backbone: sim_data_projected        # optional; omit if no attention mask is used
      backbones:
        sim_data_projected: {type: set_transformer, summary_dim: 32, num_blocks: 2, ...}
        vcirc_kms: {type: time_series_transformer, summary_dim: 32, ...}
      head: {widths: [64, 64], output_dim: 32}  # optional

Each backbone spec is a full ``SummaryNetworkConfig`` (missing fields take that schema's
defaults) resolved through the summary-network registry, so custom architectures work as
backbones too.
"""

from __future__ import annotations

from collections.abc import Mapping

import keras
from keras import ops

from bayesflow.networks.summary.summary_network import SummaryNetwork
from bayesflow.types import Shape, Tensor
from bayesflow.utils.serialization import deserialize, serializable, serialize

from hydrabflow.networks.factory import build_summary_network, register_summary_network


@serializable("hydrabflow.networks")
class MaskedFusionNetwork(SummaryNetwork):
    """Fuse one summary backbone per named input; route the attention mask to one of them."""

    def __init__(
        self,
        backbones: Mapping[str, keras.Layer],
        head: keras.Layer | None = None,
        mask_backbone: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbones = backbones
        self.head = head
        self.mask_backbone = mask_backbone
        self._ordered_keys = sorted(self.backbones.keys())

    def build(self, inputs_shape: Mapping[str, Shape]):
        if self.built:
            return
        output_shapes = []
        for k in self._ordered_keys:
            shape = inputs_shape[k]
            if not self.backbones[k].built:
                self.backbones[k].build(shape)
            output_shapes.append(self.backbones[k].compute_output_shape(shape))
        if self.head is not None and not self.head.built:
            fused = (*output_shapes[0][:-1], sum(s[-1] for s in output_shapes))
            self.head.build(fused)
        self.built = True

    def compute_output_shape(self, inputs_shape: Mapping[str, Shape]):
        output_shapes = [
            self.backbones[k].compute_output_shape(inputs_shape[k]) for k in self._ordered_keys
        ]
        out = (*output_shapes[0][:-1], sum(s[-1] for s in output_shapes))
        if self.head is not None:
            out = self.head.compute_output_shape(out)
        return out

    def _backbone_kwargs(self, key: str, attention_mask: Tensor | None) -> dict:
        if attention_mask is not None and key == self.mask_backbone:
            return {"attention_mask": attention_mask}
        return {}

    def call(
        self,
        inputs: Mapping[str, Tensor],
        attention_mask: Tensor = None,
        training: bool = False,
    ) -> Tensor:
        outputs = [
            self.backbones[k](
                inputs[k], training=training, **self._backbone_kwargs(k, attention_mask)
            )
            for k in self._ordered_keys
        ]
        fused = ops.concatenate(outputs, axis=-1)
        if self.head is None:
            return fused
        return self.head(fused, training=training)

    def compute_metrics(
        self,
        inputs: Mapping[str, Tensor],
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        if not self.built:
            self.build(keras.tree.map_structure(keras.ops.shape, inputs))

        attention_mask = kwargs.get("attention_mask")
        is_training = stage == "training"
        metrics: dict[str, list] = {"loss": [], "outputs": []}

        for k in self._ordered_keys:
            backbone = self.backbones[k]
            extra = self._backbone_kwargs(k, attention_mask)
            if isinstance(backbone, SummaryNetwork):
                metrics_k = backbone.compute_metrics(inputs[k], stage=stage, **extra)
                metrics["outputs"].append(metrics_k["outputs"])
                if "loss" in metrics_k:
                    metrics["loss"].append(metrics_k["loss"])
            else:
                metrics["outputs"].append(backbone(inputs[k], training=is_training, **extra))

        if metrics["loss"]:
            metrics["loss"] = ops.sum(metrics["loss"])
        else:
            del metrics["loss"]

        metrics["outputs"] = ops.concatenate(metrics["outputs"], axis=-1)
        if self.head is not None:
            metrics["outputs"] = self.head(metrics["outputs"], training=is_training)
        return metrics

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "backbones": self.backbones,
            "head": self.head,
            "mask_backbone": self.mask_backbone,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@register_summary_network("fusion")
def _fusion(cfg):
    """Build a :class:`MaskedFusionNetwork` from ``cfg.params`` (see module docstring)."""
    import bayesflow as bf
    from omegaconf import OmegaConf

    from hydrabflow.config.schema import SummaryNetworkConfig

    params = (
        OmegaConf.to_container(cfg.params, resolve=True)
        if OmegaConf.is_config(cfg.params)
        else dict(cfg.params)
    )
    backbone_specs = params.get("backbones")
    if not backbone_specs:
        raise ValueError(
            "summary_network.type=fusion needs params.backbones: "
            "{<observable key>: {type: <summary net>, ...}, ...}"
        )

    backbones = {}
    for key, spec in backbone_specs.items():
        # Overlay the spec on the schema so unspecified fields keep their defaults.
        merged = OmegaConf.merge(OmegaConf.structured(SummaryNetworkConfig), spec or {})
        backbones[key] = build_summary_network(merged)

    head = None
    head_spec = params.get("head")
    if head_spec:
        # `widths` is an explicit list; `width` + `depth` scalars are the tunable alternative
        # (Optuna search spaces address scalar config fields, not lists).
        if "width" in head_spec or "depth" in head_spec:
            widths = [int(head_spec.get("width", 64))] * int(head_spec.get("depth", 2))
        else:
            widths = [int(w) for w in head_spec.get("widths", [64, 64])]
        head = keras.Sequential(
            [
                bf.networks.MLP(widths=widths),
                keras.layers.Dense(units=int(head_spec.get("output_dim", 32))),
            ]
        )

    return MaskedFusionNetwork(
        backbones=backbones,
        head=head,
        mask_backbone=params.get("mask_backbone"),
    )
