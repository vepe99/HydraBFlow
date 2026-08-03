"""Build BayesFlow networks from the typed config, resolved by ``cfg.type``.

A custom architecture plugs in without touching this file: drop a module into
``src/hydrabflow/networks/`` with an ``@register_summary_network("my_net")`` (or
``@register_inference_network``) decorated builder — the package auto-imports it — and select it with
``model.summary_network.type=my_net``. Custom builders read extras from ``cfg.params``.

Multi-observable **fusion** is the extension seam: when the adapter groups several observable keys
into ``summary_variables``, build one net per key and combine them with
``bayesflow.networks.FusionNetwork``. ``bayesflow`` is imported lazily so config-only contexts (and
most of the test suite) don't need the backend.
"""

from __future__ import annotations

from typing import Any, Callable

from hydrabflow.utils.registry import Registry

Builder = Callable[[Any], Any]  # network config -> BayesFlow/keras network

SUMMARY_NETWORKS: Registry[Builder] = Registry("summary_network.type")
INFERENCE_NETWORKS: Registry[Builder] = Registry("inference_network.type")

register_summary_network = SUMMARY_NETWORKS.add
register_inference_network = INFERENCE_NETWORKS.add


def build_summary_network(cfg) -> Any:
    """Return the summary network selected by ``cfg.type``."""
    return SUMMARY_NETWORKS.get(cfg.type)(cfg)


def build_inference_network(cfg) -> Any:
    """Return the inference (posterior) network selected by ``cfg.type``."""
    return INFERENCE_NETWORKS.get(cfg.type)(cfg)


# --------------------------------------------------------------------------------------------- #
# Shipped builders
# --------------------------------------------------------------------------------------------- #


def _embed_dim(cfg) -> int:
    """Attention width. Expressed per head, so ``embed_dim % num_heads == 0`` always holds — a
    tuner sampling the width and the head count independently can never draw an invalid pair."""
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
