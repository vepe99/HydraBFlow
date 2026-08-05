"""The shipped network builders. A builder maps a network config to a BayesFlow network.

To add your own, drop a module in this package and decorate a function:

    @register_summary_network("my_net")
    def _my_net(cfg):
        return MyNetwork(summary_dim=cfg.summary_dim, **cfg.params)

Then select it with ``model.summary_network.type=my_net``. Extra knobs go in ``cfg.params``.
``bayesflow`` is imported inside each builder so config-only contexts don't need the backend.
"""

from __future__ import annotations

from typing import Any

from hydrabflow.registry import register_inference_network, register_summary_network


def _embed_dim(cfg) -> int:
    """Attention width, expressed per head so ``embed_dim % num_heads == 0`` always holds."""
    return int(cfg.num_heads) * int(cfg.embed_dim_per_head)


@register_summary_network("set_transformer")
def _set_transformer(cfg) -> Any:
    import bayesflow as bf

    blocks = int(cfg.num_blocks)
    return bf.networks.SetTransformer(
        summary_dim=int(cfg.summary_dim),
        embed_dims=(_embed_dim(cfg),) * blocks,
        num_heads=(int(cfg.num_heads),) * blocks,
        mlp_depths=(int(cfg.mlp_depth),) * blocks,
        mlp_widths=(int(cfg.mlp_width),) * blocks,
        dropout=float(cfg.dropout),
    )


@register_summary_network("time_series_transformer")
def _time_series_transformer(cfg) -> Any:
    import bayesflow as bf

    blocks = int(cfg.num_blocks)
    return bf.networks.TimeSeriesTransformer(
        summary_dim=int(cfg.summary_dim),
        embed_dims=(_embed_dim(cfg),) * blocks,
        num_heads=(int(cfg.num_heads),) * blocks,
        dropout=float(cfg.dropout),
        # params.time_axis: index of a time channel carried inside the observation (e.g. -1 when a
        # time column was concatenated onto the values); None = evenly spaced steps.
        time_axis=cfg.params.get("time_axis"),
    )


@register_summary_network("deep_set")
def _deep_set(cfg) -> Any:
    import bayesflow as bf

    return bf.networks.DeepSet(summary_dim=int(cfg.summary_dim), dropout=float(cfg.dropout))


@register_inference_network("flow_matching")
def _flow_matching(cfg) -> Any:
    import bayesflow as bf

    return bf.networks.FlowMatching(
        subnet_kwargs={
            "widths": [int(cfg.mlp_width)] * int(cfg.mlp_depth),
            "dropout": float(cfg.dropout),
        },
    )


@register_inference_network("diffusion")
def _diffusion(cfg) -> Any:
    import bayesflow as bf

    return bf.networks.DiffusionModel(
        subnet_kwargs={
            "widths": [int(cfg.mlp_width)] * int(cfg.mlp_depth),
            "time_embedding_dim": int(cfg.time_embedding_dim),
        },
    )
