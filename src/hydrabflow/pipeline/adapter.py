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

    With compositional score modeling (``composition.level``), the same derivation targets one
    level of the simulator's hierarchy:
      * ``global`` — infer ``global_parameter_names``, conditioned on ``context_keys``;
      * ``local`` — infer ``local_parameter_names``, conditioned on the global parameters +
        ``context_keys``.
    """
    from hydrabflow.simulators.registry import get_simulator

    needs_inference = not _as_list(cfg.adapter.inference_variables)
    needs_summary = not _as_list(cfg.adapter.summary_variables)
    needs_conditions = not _as_list(cfg.adapter.inference_conditions)
    if not (needs_inference or needs_summary or needs_conditions):
        return
    try:
        simulator = get_simulator(cfg.simulator)
    except KeyError:
        return  # no registered simulator: adapter must be configured explicitly

    level = str(getattr(getattr(cfg, "composition", None), "level", "none") or "none")
    if level == "global":
        inference_variables = simulator.global_parameter_names
        inference_conditions = simulator.context_keys
    elif level == "local":
        inference_variables = simulator.local_parameter_names
        inference_conditions = simulator.global_parameter_names + simulator.context_keys
    else:
        inference_variables = simulator.parameter_names
        inference_conditions = []

    if needs_inference:
        cfg.adapter.inference_variables = list(inference_variables)
    if needs_summary:
        cfg.adapter.summary_variables = list(simulator.observable_keys)
    if needs_conditions and inference_conditions:
        cfg.adapter.inference_conditions = list(inference_conditions)


def adapter_keys(cfg) -> List[str]:
    """All dataset keys the adapter (``cfg.adapter``) consumes, in a stable order."""
    keys = (
        _as_list(cfg.adapter.inference_variables)
        + _as_list(cfg.adapter.summary_variables)
        + _as_list(cfg.adapter.inference_conditions)
    )
    mask_key = getattr(cfg.adapter, "attention_mask_key", None)
    if mask_key:
        keys.append(str(mask_key))
    return keys


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

    # Boolean per-particle mask (e.g. from an observational-window augmentation) renamed to the
    # role BayesFlow's approximator forwards to the summary network as `attention_mask`.
    mask_key = getattr(cfg, "attention_mask_key", None)
    if mask_key:
        adapter = adapter.rename(str(mask_key), "summary_attention_mask")

    if len(summary_variables) == 1:
        adapter = adapter.rename(summary_variables[0], "summary_variables")
    elif len(summary_variables) > 1:
        # Fusion: group multiple observables into summary_variables; the `fusion` summary network
        # (networks/fusion.py) builds one backbone per key to consume them.
        adapter = adapter.group(summary_variables, into="summary_variables")

    if inference_conditions:
        adapter = adapter.concatenate(inference_conditions, into="inference_conditions")

    return adapter
