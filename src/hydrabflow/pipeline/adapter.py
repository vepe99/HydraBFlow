"""Build the BayesFlow ``Adapter``: dataset keys -> the roles BayesFlow expects.

``inference_variables`` is the target, ``summary_variables`` feeds the summary network, and
``inference_conditions`` are passed to the inference network directly. One summary key is renamed to
the role; several are ``group``ed, which is what ``FusionNetwork`` consumes (see
``registry.build_summary_network``).
"""

from __future__ import annotations

import logging
from typing import Any, List

log = logging.getLogger(__name__)


def _lists(cfg) -> tuple[List[str], List[str], List[str], List[str]]:
    """The four adapter key lists as plain lists (resolving interpolations)."""
    from omegaconf import OmegaConf

    c = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    return (
        list(c["inference_variables"]),
        list(c["summary_variables"]),
        list(c["inference_conditions"]),
        list(c["drop"]),
    )


def fill_adapter_from_simulator(cfg) -> None:
    """Fill empty adapter variable lists from the simulator's own declaration (in place).

    Explicit config values win — the escape hatch for data no registered simulator produced.
    """
    from hydrabflow.registry import get_simulator

    needs_inference = not list(cfg.adapter.inference_variables)
    needs_summary = not list(cfg.adapter.summary_variables)
    if not (needs_inference or needs_summary):
        return
    try:
        simulator = get_simulator(cfg.simulator)
    except KeyError:
        return  # no registered simulator: the adapter must be configured explicitly
    if needs_inference:
        cfg.adapter.inference_variables = list(simulator.parameter_names)
    if needs_summary:
        cfg.adapter.summary_variables = list(simulator.observable_keys)


def adapter_keys(cfg) -> List[str]:
    """Every dataset key the adapter consumes.

    ``drop`` is included so dropped keys still survive :func:`select_adapter_keys`: a dropped key may
    be an augmentation *input* that must reach the augmentation chain before ``.drop()`` removes it.
    """
    inference, summary, conditions, drop = _lists(cfg.adapter)
    return inference + summary + conditions + drop


def select_adapter_keys(data: dict, cfg) -> dict:
    """Keep only the dataset keys the adapter consumes (simulators write extra arrays)."""
    wanted = set(adapter_keys(cfg))
    dropped = [k for k in data if k not in wanted]
    if dropped:
        log.info("Dropping keys the adapter does not use: %s", dropped)
    return {k: v for k, v in data.items() if k in wanted}


def build_adapter(cfg) -> Any:
    """Construct ``bf.adapters.Adapter`` from an ``AdapterConfig``."""
    import bayesflow as bf

    inference_variables, summary_variables, inference_conditions, drop = _lists(cfg)

    if not inference_variables:
        raise ValueError(
            "adapter.inference_variables is empty and could not be derived from the simulator. "
            "Either select a registered simulator (they declare parameter_names/observable_keys) "
            "or set adapter.inference_variables / adapter.summary_variables in conf/config.yaml "
            "(see docs/running.md for the no-simulator workflow)."
        )

    # create_default = to_array + convert_dtype(float64->float32) + concatenate(into inference_variables)
    adapter = bf.adapters.Adapter.create_default(inference_variables)
    if drop:
        adapter = adapter.drop(drop)
    if len(summary_variables) == 1:
        adapter = adapter.rename(summary_variables[0], "summary_variables")
    elif len(summary_variables) > 1:
        adapter = adapter.group(summary_variables, into="summary_variables")  # fusion
    if inference_conditions:
        adapter = adapter.concatenate(inference_conditions, into="inference_conditions")
    return adapter
