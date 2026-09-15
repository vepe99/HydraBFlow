"""Occupancy-aware TimeSeriesTransformer for the φ1-gridded stream summary statistics.

``stream_summary_grid`` lays the per-stream summaries out as a φ1 time series and substitutes 0 for
bins that hold too few members to estimate a median/dispersion. A stock
:class:`bayesflow.networks.TimeSeriesTransformer` cannot tell those apart from real measurements —
and for φ2 a fabricated 0 *is* the on-track value, so an empty bin reads as a perfectly on-track,
zero-dispersion one. This wrapper consumes the occupancy channels the augmentation now exports and
removes those bins from the summary:

1. read the per-bin member counts from ``count_channel`` (astrometric tracks) and, when given,
   ``vlos_count_channel`` (v_los uses measured stars only, so it empties out sooner);
2. hard-zero the statistic channels of under-populated bins, so a substituted 0 never masquerades
   as data, **and hard-zero the occupancy channels themselves** — the counts are a validity signal,
   not an observable, so the network must not train on their magnitudes (a simulated bin's member
   count reflects the progenitor draw, not anything the real catalogue measures);
3. forward a ``(batch, 1, 1, K)`` ``attention_mask`` so those bins are excluded from attention;
4. **pool only over the valid bins.** This step is the reason the wrapper exists: BayesFlow's
   ``TimeSeriesTransformer.call`` ends with ``summary = self.pooling(inp)`` and its
   ``SetTransformer.call`` with ``self.pooling_by_attention(x, training=training)`` — neither
   receives ``attention_mask``, so masked positions still leak into the summary (measured at
   20–35% of the output scale). We therefore run the inner network with ``return_sequences=True``
   and do the pooling here, over the valid bins only.

Because pooling is masked, a learned "missing bin" token would contribute nothing, so there is
none — unlike :mod:`hydrabflow.networks.masked_set_transformer`, where the masked elements are real
stars that must stay in the set.

Selected via ``model/summary_network`` (or as a fusion backbone spec)::

    type: masked_time_series_transformer
    summary_dim: 32
    num_blocks: 2
    num_heads: 4
    params:
      time_axis: -1              # φ1 bin centre, last channel
      count_channel: 10          # n_track   (14-channel stream_summary_grid layout)
      vlos_count_channel: 11     # n_vlos
      stat_channels: [0,1,2,3,4,5,6,7]    # the astrometric median/std channels
      vlos_stat_channels: [8,9]           # the v_los median/std channels
      min_count: 3
"""

from __future__ import annotations

import numpy as np

import keras
from keras import ops

from bayesflow.networks.summary.summary_network import SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils.serialization import deserialize, serializable, serialize

from hydrabflow.networks.factory import _embed_dim
from hydrabflow.registry import register_summary_network


@serializable("hydrabflow.networks")
class MaskedTimeSeriesTransformer(SummaryNetwork):
    """Zero and mask under-populated φ1 bins, then pool a TimeSeriesTransformer over the rest."""

    def __init__(
        self,
        time_series_transformer: keras.Layer,
        count_channel: int,
        stat_channels: list[int],
        vlos_count_channel: int | None = None,
        vlos_stat_channels: list[int] | None = None,
        min_count: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.time_series_transformer = time_series_transformer
        self.count_channel = int(count_channel)
        self.stat_channels = [int(c) for c in stat_channels]
        self.vlos_count_channel = None if vlos_count_channel is None else int(vlos_count_channel)
        self.vlos_stat_channels = [int(c) for c in (vlos_stat_channels or [])]
        self.min_count = int(min_count)
        self._keep_track = None  # (channels,) 0/1 selectors, set in build
        self._keep_vlos = None
        self._keep_counts = None

    def _selector(self, n_channels: int, zeroed: list[int]) -> np.ndarray:
        keep = np.ones((n_channels,), dtype="float32")
        keep[zeroed] = 0.0
        return keep

    def build(self, input_shape):
        if self.built:
            return
        n_channels = int(input_shape[-1])
        referenced = [self.count_channel, *self.stat_channels, *self.vlos_stat_channels]
        if self.vlos_count_channel is not None:
            referenced.append(self.vlos_count_channel)
        bad = [c for c in referenced if not 0 <= c < n_channels]
        if bad:
            raise ValueError(
                f"masked_time_series_transformer channel indices {bad} out of range for input with "
                f"{n_channels} feature channels — count/stat channels must match the "
                "stream_summary_grid layout (see its docstring)."
            )
        self._keep_track = self._selector(n_channels, self.stat_channels)
        self._keep_vlos = self._selector(n_channels, self.vlos_stat_channels)
        counts = [self.count_channel]
        if self.vlos_count_channel is not None:
            counts.append(self.vlos_count_channel)
        self._keep_counts = self._selector(n_channels, counts)
        if not self.time_series_transformer.built:
            self.time_series_transformer.build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        # inner runs with return_sequences=True -> (batch, K, summary_dim); we pool the K axis away
        seq = self.time_series_transformer.compute_output_shape(input_shape)
        return (*seq[:-2], seq[-1])

    def _valid(self, x: Tensor, channel: int) -> Tensor:
        """(batch, K, 1) float mask: 1 where the bin has at least ``min_count`` members."""
        count = x[..., channel : channel + 1]
        return ops.cast(count >= float(self.min_count), x.dtype)

    def _prepare(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Zero the statistics of under-populated bins and then the occupancy channels themselves.

        Returns the prepared grid and the ``(b, K, 1)`` validity mask the counts encoded, so a
        subclass can reuse the prep without repeating the channel bookkeeping.
        """
        valid = self._valid(x, self.count_channel)  # (b, K, 1)

        keep = ops.cast(ops.convert_to_tensor(self._keep_track), x.dtype)
        x = x * (keep + (1.0 - keep) * valid)
        if self.vlos_count_channel is not None and self.vlos_stat_channels:
            valid_v = self._valid(x, self.vlos_count_channel)
            keep_v = ops.cast(ops.convert_to_tensor(self._keep_vlos), x.dtype)
            x = x * (keep_v + (1.0 - keep_v) * valid_v)

        # The counts have now done their job (validity); drop their values so the network never
        # sees a member count as a feature.
        return x * ops.cast(ops.convert_to_tensor(self._keep_counts), x.dtype), valid

    def call(self, x: Tensor, training: bool = False, attention_mask: Tensor = None) -> Tensor:
        x, valid = self._prepare(x)

        # Exclude invalid bins from attention. Shape (b, 1, 1, K) broadcasts over heads and queries.
        bin_mask = ops.cast(valid[..., 0], "bool")  # (b, K)
        mask = bin_mask[:, None, None, :]
        if attention_mask is not None:
            # An externally supplied mask (same (b, K) bin geometry) is intersected, not replaced.
            outer = ops.cast(attention_mask, "bool")
            if len(ops.shape(outer)) == 2:
                outer = outer[:, None, None, :]
            mask = ops.logical_and(mask, outer)

        seq = self.time_series_transformer(x, training=training, attention_mask=mask)

        # Masked mean pooling over the valid bins. Rows with no valid bin at all fall back to a
        # plain mean so the output stays finite instead of turning into 0/0.
        w = valid  # (b, K, 1)
        total = ops.sum(w, axis=1)  # (b, 1)
        any_valid = ops.cast(total > 0.0, x.dtype)
        pooled = ops.sum(seq * w, axis=1) / ops.maximum(total, 1.0)
        fallback = ops.mean(seq, axis=1)
        return any_valid * pooled + (1.0 - any_valid) * fallback

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "time_series_transformer": self.time_series_transformer,
            "count_channel": self.count_channel,
            "stat_channels": self.stat_channels,
            "vlos_count_channel": self.vlos_count_channel,
            "vlos_stat_channels": self.vlos_stat_channels,
            "min_count": self.min_count,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@register_summary_network("masked_time_series_transformer")
def _masked_time_series_transformer(cfg):
    """Build a :class:`MaskedTimeSeriesTransformer` from a ``SummaryNetworkConfig``.

    Channel defaults match the 14-channel ``stream_summary_grid`` layout
    ``[med/std x 4 astrometric, med/std vlos, n_track, n_vlos, j, φ1_centre]``.
    """
    import bayesflow as bf

    blocks = int(cfg.num_blocks)
    embed_dim = _embed_dim(cfg)
    params = cfg.params or {}
    inner = bf.networks.TimeSeriesTransformer(
        summary_dim=int(cfg.summary_dim),
        embed_dims=(embed_dim,) * blocks,
        num_heads=(int(cfg.num_heads),) * blocks,
        dropout=float(cfg.dropout),
        time_axis=int(params.get("time_axis", -1)),
        return_sequences=True,  # pooling happens in the wrapper, masked
    )
    return MaskedTimeSeriesTransformer(
        time_series_transformer=inner,
        count_channel=int(params.get("count_channel", 10)),
        stat_channels=[int(c) for c in params.get("stat_channels", [0, 1, 2, 3, 4, 5, 6, 7])],
        vlos_count_channel=(
            None
            if params.get("vlos_count_channel", 11) is None
            else int(params.get("vlos_count_channel", 11))
        ),
        vlos_stat_channels=[int(c) for c in params.get("vlos_stat_channels", [8, 9])],
        min_count=int(params.get("min_count", 3)),
    )


@serializable("hydrabflow.networks")
class MaskedMLP(MaskedTimeSeriesTransformer):
    """The same occupancy prep as :class:`MaskedTimeSeriesTransformer`, feeding a flat MLP.

    Under-populated φ1 bins have their statistic channels hard-zeroed and the occupancy channels
    are zeroed once they have done their job, exactly as in the transformer — so the two backbones
    differ only in the architecture that consumes the prepared grid, which is what makes an
    MLP-vs-transformer ablation a clean one. There is no attention or pooling to mask here: the
    grid is flattened, so an empty bin contributes a block of zeros at a fixed position and the
    network can learn what that means.
    """

    def compute_output_shape(self, input_shape):
        return self.time_series_transformer.compute_output_shape(input_shape)

    def call(self, x: Tensor, training: bool = False, attention_mask: Tensor = None) -> Tensor:
        prepared, _ = self._prepare(x)
        return self.time_series_transformer(prepared, training=training)


@register_summary_network("masked_mlp")
def _masked_mlp(cfg):
    """MLP counterpart of ``masked_time_series_transformer`` — same ``params`` channel contract."""
    import bayesflow as bf
    import keras

    params = cfg.params or {}
    inner = keras.Sequential(
        [
            keras.layers.Flatten(),
            bf.networks.MLP(
                widths=[int(cfg.mlp_width)] * int(cfg.mlp_depth), dropout=float(cfg.dropout)
            ),
            keras.layers.Dense(units=int(cfg.summary_dim)),
        ]
    )
    return MaskedMLP(
        time_series_transformer=inner,
        count_channel=int(params.get("count_channel", 10)),
        stat_channels=[int(c) for c in params.get("stat_channels", [0, 1, 2, 3, 4, 5, 6, 7])],
        vlos_count_channel=(
            None
            if params.get("vlos_count_channel", 11) is None
            else int(params.get("vlos_count_channel", 11))
        ),
        vlos_stat_channels=[int(c) for c in params.get("vlos_stat_channels", [8, 9])],
        min_count=int(params.get("min_count", 3)),
    )
