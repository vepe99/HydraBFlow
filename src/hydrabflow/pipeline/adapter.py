"""Build the BayesFlow ``Adapter`` from ``AdapterConfig``.

The adapter is the structural (non-stochastic) transform that maps raw dataset keys to the roles
BayesFlow expects: ``inference_variables`` (the target), ``summary_variables`` (fed to the
summary network), and ``inference_conditions`` (direct conditions). Generalizes the reference's
hand-written adapter chain (main_train_new_rotationcurve_agama.py:108-122).

Single observable (default): the one ``summary_variables`` key is renamed to the BayesFlow role.
Fusion seam: with multiple keys, ``group`` them into ``summary_variables`` and build one summary
backbone per key in ``networks.factory`` (left commented below — uncomment + adjust to enable).
"""

from __future__ import annotations

from typing import Any, List


def _as_list(x) -> List[str]:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(x):
        return list(OmegaConf.to_container(x, resolve=True))
    return list(x)


def fill_adapter_from_simulator(cfg) -> None:
    """Fill empty adapter variable lists from the simulator's own declaration (in place).

    The simulator class is the single source of truth for its parameter names and observable
    keys, so by default the adapter derives ``inference_variables`` / ``summary_variables`` from
    it and the user never repeats them in config. Explicit (non-empty) config values win — that
    is the escape hatch for datasets not produced by a registered simulator (bring-your-own-data),
    where no simulator may exist: in that case the lists are left empty here and
    :func:`build_adapter` raises with instructions.

    Observables listed in ``adapter.drop`` are **not** derived: dropping an observable is how an
    unconditional (prior-only) approximator is configured, and re-deriving it from the simulator
    would silently put it back (see ``conf/adapter/lv_unconditional.yaml``).
    """
    from hydrabflow.simulators.registry import get_simulator

    needs_inference = not _as_list(cfg.adapter.inference_variables)
    needs_summary = not _as_list(cfg.adapter.summary_variables)
    if not (needs_inference or needs_summary):
        return
    try:
        simulator = get_simulator(cfg.simulator)
    except KeyError:
        return  # no registered simulator: adapter must be configured explicitly
    if needs_inference:
        cfg.adapter.inference_variables = list(simulator.parameter_names)
    if needs_summary:
        dropped = set(_as_list(cfg.adapter.drop))
        cfg.adapter.summary_variables = [
            key for key in simulator.observable_keys if key not in dropped
        ]


def adapter_keys(cfg) -> List[str]:
    """All dataset keys the adapter (``cfg.adapter``) consumes, in a stable order.

    ``drop`` is included so keys the adapter drops still survive :func:`select_adapter_keys`: a
    dropped key may be a *per-batch augmentation input* that the network must NOT see — it has to
    reach the augmentation chain, then the adapter's ``.drop()`` removes it before the approximator.
    """
    return (
        _as_list(cfg.adapter.inference_variables)
        + _as_list(cfg.adapter.summary_variables)
        + _as_list(cfg.adapter.inference_conditions)
        + _as_list(cfg.adapter.drop)
    )


def select_adapter_keys(data: dict, cfg) -> dict:
    """Keep only the dataset keys the adapter consumes (the generalized ``keys_to_drop``).

    Simulator datasets carry extra arrays (fixed constants, intermediate coordinates) that the
    model must not see; BayesFlow would otherwise pass them through to the approximator.
    Keys created later, per batch, by augmentations are unaffected.
    """
    wanted = set(adapter_keys(cfg))
    dropped = [k for k in data if k not in wanted]
    if dropped:
        from hydrabflow.utils.logging import get_logger

        get_logger(__name__).info("Dropping dataset keys the adapter does not use: %s", dropped)
    return {k: v for k, v in data.items() if k in wanted}


def build_adapter(cfg) -> Any:
    """Construct ``bf.adapters.Adapter`` from ``cfg`` (an ``AdapterConfig``)."""
    import bayesflow as bf

    inference_variables = _as_list(cfg.inference_variables)
    summary_variables = _as_list(cfg.summary_variables)

    if not inference_variables:
        raise ValueError(
            "adapter.inference_variables is empty and could not be derived from the simulator. "
            "Either select a registered simulator (they declare parameter_names/observable_keys) "
            "or set adapter.inference_variables / adapter.summary_variables explicitly "
            "(see conf/adapter/two_moons.yaml for an explicit example and "
            "docs/bring_your_own_data.md for the no-simulator workflow)."
        )
    inference_conditions = _as_list(cfg.inference_conditions)
    drop = _as_list(cfg.drop)

    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(inference_variables, into="inference_variables")
    )

    if drop:
        adapter = adapter.drop(drop)

    if len(summary_variables) == 1:
        adapter = adapter.rename(summary_variables[0], "summary_variables")
    elif len(summary_variables) > 1:
        # --- Fusion seam -------------------------------------------------------------------- #
        # Group multiple observables into summary_variables; build a FusionNetwork (one backbone
        # per key) in networks.factory to consume them.
        adapter = adapter.group(summary_variables, into="summary_variables")
        # ------------------------------------------------------------------------------------ #

    if inference_conditions:
        adapter = adapter.concatenate(inference_conditions, into="inference_conditions")

    return adapter
