"""Missingness-aware SetTransformer for stars without a measured line-of-sight velocity.

The stream observation model marks unmeasured v_los with a binary indicator channel
(``concatenate_vlos_mask``) but the imputed value/sigma channels still carry a fill that the
stock SetTransformer must learn to distrust. This wrapper makes the missingness explicit at the
architecture level (NAIM-style / BERT mask token):

1. read the per-star measured/missing flag from its feature channel (``mask_channel``);
2. hard-zero the v_los value and sigma channels (``value_channels`` / ``sigma_channels``) where
   the flag is 0 — so whatever fill upstream produced (per-stream mean in the shipped real npz,
   0.0 with ``vlos_impute: zero``) never reaches the attention stack;
3. project each star through a shared ``Dense(embed_dim)`` and **add a learned missing-v_los
   embedding vector** to stars without a measurement;
4. run a stock :class:`bayesflow.networks.SetTransformer` on the embeddings, forwarding the
   set-level ``attention_mask`` (window/padding mask) unchanged.

Selected via ``model/summary_network`` (or as a fusion backbone spec)::

    type: masked_set_transformer
    summary_dim: 36
    num_blocks: 4
    num_heads: 3
    params:
      embed_dim_multiplier: 14   # or embed_dim
      value_channels: [5]        # v_los value  (stream_global 15-channel layout)
      sigma_channels: [11]       # v_los sigma
      mask_channel: 13           # 0/1 indicator, 1 = measured
"""

from __future__ import annotations

import numpy as np

import keras
from keras import ops

from bayesflow.networks.summary.summary_network import SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils.serialization import deserialize, serializable, serialize

from hydrabflow.networks.factory import _embed_dim, register_summary_network


@serializable("hydrabflow.networks")
class MaskedSetTransformer(SummaryNetwork):
    """Zero masked feature channels, add a learned missing-value embedding, run a SetTransformer."""

    def __init__(
        self,
        set_transformer: keras.Layer,
        embed_dim: int,
        value_channels: list[int],
        sigma_channels: list[int],
        mask_channel: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_transformer = set_transformer
        self.embed_dim = int(embed_dim)
        self.value_channels = [int(c) for c in value_channels]
        self.sigma_channels = [int(c) for c in sigma_channels]
        self.mask_channel = int(mask_channel)
        self.projector = keras.layers.Dense(self.embed_dim, name="feature_projector")
        self._keep = None  # (channels,) 0/1 selector, set in build

    def build(self, input_shape):
        if self.built:
            return
        n_channels = int(input_shape[-1])
        zeroed = self.value_channels + self.sigma_channels
        bad = [c for c in zeroed + [self.mask_channel] if not 0 <= c < n_channels]
        if bad:
            raise ValueError(
                f"masked_set_transformer channel indices {bad} out of range for input with "
                f"{n_channels} feature channels — value/sigma/mask_channel must match the "
                "augmentation chain's concatenation layout."
            )
        keep = np.ones((n_channels,), dtype="float32")
        keep[zeroed] = 0.0
        self._keep = keep

        self.missing_token = self.add_weight(
            shape=(self.embed_dim,),
            initializer="zeros",
            trainable=True,
            name="missing_vlos_token",
        )
        self.projector.build(input_shape)
        embedded = (*input_shape[:-1], self.embed_dim)
        if not self.set_transformer.built:
            self.set_transformer.build(embedded)
        self.built = True

    def compute_output_shape(self, input_shape):
        return self.set_transformer.compute_output_shape((*input_shape[:-1], self.embed_dim))

    def call(self, x: Tensor, training: bool = False, attention_mask: Tensor = None) -> Tensor:
        measured = x[..., self.mask_channel : self.mask_channel + 1]  # (b, set, 1), 1 = measured
        keep = ops.cast(ops.convert_to_tensor(self._keep), x.dtype)
        # keep=1 channels pass through; keep=0 channels are zeroed for unmeasured stars.
        x = x * (keep + (1.0 - keep) * measured)
        h = self.projector(x)
        h = h + (1.0 - measured) * self.missing_token
        return self.set_transformer(h, training=training, attention_mask=attention_mask)

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "set_transformer": self.set_transformer,
            "embed_dim": self.embed_dim,
            "value_channels": self.value_channels,
            "sigma_channels": self.sigma_channels,
            "mask_channel": self.mask_channel,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@register_summary_network("masked_set_transformer")
def _masked_set_transformer(cfg):
    """Build a :class:`MaskedSetTransformer` from a ``SummaryNetworkConfig`` (see module docstring)."""
    import bayesflow as bf

    blocks = int(cfg.num_blocks)
    embed_dim = _embed_dim(cfg)
    params = cfg.params or {}
    inner = bf.networks.SetTransformer(
        summary_dim=int(cfg.summary_dim),
        embed_dims=(embed_dim,) * blocks,
        num_heads=(int(cfg.num_heads),) * blocks,
        mlp_depths=(int(cfg.mlp_depth),) * blocks,
        mlp_widths=(int(cfg.mlp_width),) * blocks,
        dropout=float(cfg.dropout),
    )
    return MaskedSetTransformer(
        set_transformer=inner,
        embed_dim=embed_dim,
        value_channels=[int(c) for c in params.get("value_channels", [5])],
        sigma_channels=[int(c) for c in params.get("sigma_channels", [11])],
        mask_channel=int(params.get("mask_channel", 13)),
    )
