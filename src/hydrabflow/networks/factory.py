"""Build BayesFlow networks from structured dataclass configs (no ``_target_``).

Builders are resolved by ``cfg.type`` through name -> builder registries, so a custom
architecture plugs in without touching this file: drop a module into ``src/hydrabflow/networks/``
with an ``@register_summary_network("my_net")`` (or ``@register_inference_network``) decorated
builder — the package auto-imports it — and select it with ``model/summary_network.type=my_net``.
Custom builders can read free-form extras from ``cfg.params``.

The shipped builders are thin, opinionated wrappers around ``bayesflow.networks``. They translate
the scalar hyperparameters in ``SummaryNetworkConfig`` / ``InferenceNetworkConfig`` into the
constructor kwargs each network expects (e.g. expanding ``num_blocks`` into per-block tuples).
``bayesflow`` is imported lazily so config-only contexts (and the test suite) don't require the
backend.

Multi-observable **fusion** is the documented extension seam: when an adapter groups several
observable keys into ``summary_variables``, build one summary net per key and combine them with
``bayesflow.networks.FusionNetwork``. The default single-observable path returns one net.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

Builder = Callable[[Any], Any]  # network config dataclass -> BayesFlow/keras network

_SUMMARY_BUILDERS: Dict[str, Builder] = {}
_INFERENCE_BUILDERS: Dict[str, Builder] = {}


def register_summary_network(name: str):
    """Decorator registering a summary-network builder under ``name`` (the config ``type``)."""

    def _wrap(fn: Builder) -> Builder:
        _SUMMARY_BUILDERS[name] = fn
        return fn

    return _wrap


def register_inference_network(name: str):
    """Decorator registering an inference-network builder under ``name`` (the config ``type``)."""

    def _wrap(fn: Builder) -> Builder:
        _INFERENCE_BUILDERS[name] = fn
        return fn

    return _wrap


def build_summary_network(cfg) -> Any:
    """Return the summary network selected by ``cfg.type`` (a ``SummaryNetworkConfig``)."""
    if cfg.type not in _SUMMARY_BUILDERS:
        raise ValueError(
            f"Unknown summary_network.type '{cfg.type}'. Available: {sorted(_SUMMARY_BUILDERS)}. "
            "Register custom architectures with @register_summary_network in "
            "src/hydrabflow/networks/."
        )
    return _SUMMARY_BUILDERS[cfg.type](cfg)


def build_inference_network(cfg) -> Any:
    """Return the inference (posterior) network selected by ``cfg.type`` (an ``InferenceNetworkConfig``)."""
    if cfg.type not in _INFERENCE_BUILDERS:
        raise ValueError(
            f"Unknown inference_network.type '{cfg.type}'. Available: {sorted(_INFERENCE_BUILDERS)}. "
            "Register custom architectures with @register_inference_network in "
            "src/hydrabflow/networks/."
        )
    return _INFERENCE_BUILDERS[cfg.type](cfg)


# --------------------------------------------------------------------------------------------- #
# Shipped builders
# --------------------------------------------------------------------------------------------- #


@register_summary_network("set_transformer")
def _set_transformer(cfg) -> Any:
    import bayesflow as bf

    blocks = int(cfg.num_blocks)
    return bf.networks.SetTransformer(
        summary_dim=int(cfg.summary_dim),
        embed_dims=(int(cfg.embed_dim),) * blocks,
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
        embed_dims=(int(cfg.embed_dim),) * blocks,
        num_heads=(int(cfg.num_heads),) * blocks,
    )


@register_summary_network("deep_set")
def _deep_set(cfg) -> Any:
    import bayesflow as bf

    return bf.networks.DeepSet(
        summary_dim=int(cfg.summary_dim),
        dropout=float(cfg.dropout),
    )


@register_inference_network("flow_matching")
def _flow_matching(cfg) -> Any:
    import bayesflow as bf

    widths = [int(cfg.mlp_width)] * int(cfg.mlp_depth)
    return bf.networks.FlowMatching(
        subnet_kwargs={"widths": widths, "dropout": float(cfg.dropout)},
    )


@register_inference_network("diffusion")
def _diffusion(cfg) -> Any:
    import bayesflow as bf

    widths = [int(cfg.mlp_width)] * int(cfg.mlp_depth)
    return bf.networks.DiffusionModel(
        subnet_kwargs={
            "widths": widths,
            "time_embedding_dim": int(cfg.time_embedding_dim),
        },
    )
