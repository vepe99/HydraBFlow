"""Build the BayesFlow ``Adapter``: dataset keys -> the roles BayesFlow expects.

``inference_variables`` is the target, ``summary_variables`` feeds the summary network, and
``inference_conditions`` are passed to the inference network directly. One summary key is renamed to
the role; several are ``group``ed, which is what a fusion network consumes (see
``registry.build_summary_network``). ``attention_mask_key`` (stream runs) renames a per-element
boolean mask to BayesFlow's ``summary_attention_mask`` role.
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


#: ``adapter.inference_conditions: [none]`` -- an explicit "no conditions at all", as opposed to an
#: empty list, which asks `fill_adapter_from_simulator` to derive them from the simulator.
NO_CONDITIONS = "none"


def composition_level(cfg) -> str:
    """``cfg.composition.level`` as a plain string (``none`` when the block is absent)."""
    return str(getattr(getattr(cfg, "composition", None), "level", "none") or "none")


def fill_adapter_from_simulator(cfg) -> None:
    """Fill empty adapter variable lists from the simulator's own declaration (in place).

    Explicit config values win — the escape hatch for data no registered simulator produced.

    With compositional score modeling (``composition.level``) the derivation targets one level of
    the simulator's hierarchy: ``global`` infers ``global_parameter_names`` conditioned on
    ``context_keys``; ``local`` infers ``local_parameter_names`` conditioned on the globals +
    ``context_keys``.
    """
    from hydrabflow.registry import get_simulator

    inference, summary, conditions, _ = _lists(cfg.adapter)
    # An empty list means "derive it"; `[none]` means "genuinely none" -- the two-modality stream
    # adapter needs that, because its only would-be condition (the stream index j) is already a
    # channel of the sim_summary observable and must not become a condition group of its own.
    if conditions == [NO_CONDITIONS]:
        cfg.adapter.inference_conditions = []
        conditions, needs_conditions_explicitly_none = [], True
    else:
        needs_conditions_explicitly_none = False
    needs_inference, needs_summary = not inference, not summary
    needs_conditions = not conditions and not needs_conditions_explicitly_none
    if not (needs_inference or needs_summary or needs_conditions):
        return
    try:
        simulator = get_simulator(cfg.simulator)
    except KeyError:
        return  # no registered simulator: the adapter must be configured explicitly

    level = composition_level(cfg)
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


def fill_stream_grid_from_simulator(cfg) -> None:
    """Align the training-time rotation-curve grid with the simulator's ``vcirc_kms`` grid.

    The rotation-curve observable can live on a non-default radial grid (the extended Zhou u Huang
    union grid, ``simulator.params.obs_r_grid: extended``, or an explicit table,
    ``obs_r_grid: custom`` with ``obs_r_kpc``/``obs_vc_kms``/``obs_sigma_vc``). Three config nodes hardcode the default
    Zhou grid and would otherwise mismatch it: the ``mask_vcirc_radii`` preprocessing step, the
    ``attach_observed_vcirc`` step (real data) and the ``add_noise_to_vcirc`` augmentation's per-bin
    sigma. When the simulator exposes a non-default grid, inject *its* radii / sigma / observed
    curve wherever the user left them unset. No-op for every other simulator and for the default
    grid, so the plain pipeline is untouched.
    """
    from hydrabflow.registry import get_simulator

    params = getattr(cfg.simulator, "params", None)
    if str(getattr(params, "obs_r_grid", "") or "") not in ("extended", "custom"):
        return
    try:
        simulator = get_simulator(cfg.simulator)
    except KeyError:
        return
    if not hasattr(simulator, "obs_r_kpc") or not hasattr(simulator, "obs_sigma_vc"):
        return

    from omegaconf import open_dict

    radii = [float(x) for x in simulator.obs_r_kpc]
    sigma = [float(x) for x in simulator.obs_sigma_vc]
    obs_vc = [float(x) for x in simulator.obs_vc_kms] if hasattr(simulator, "obs_vc_kms") else None

    for step in getattr(getattr(cfg, "preprocessing", None), "steps", []) or []:
        # Use .get() (not getattr): DictConfig exposes dict methods as attributes, so
        # getattr(step, "values") would return the bound .values() method, not the config key.
        name = str(step.get("name", ""))
        if name == "mask_vcirc_radii" and not step.get("radii"):
            with open_dict(step):
                step.radii = radii
        elif name == "attach_observed_vcirc" and obs_vc is not None and not step.get("values"):
            with open_dict(step):
                step.values = obs_vc

    aug_params = getattr(getattr(cfg, "augmentation", None), "params", None)
    if aug_params is not None:
        with open_dict(aug_params):
            if not getattr(aug_params, "obs_r_kpc", None):
                aug_params.obs_r_kpc = radii
            if not getattr(aug_params, "obs_sigma_vc", None):
                aug_params.obs_sigma_vc = sigma


def adapter_keys(cfg) -> List[str]:
    """Every dataset key the adapter consumes.

    ``drop`` is included so dropped keys still survive :func:`select_adapter_keys`: a dropped key may
    be an augmentation *input* (e.g. the raw star cloud a summary-statistics augmentation reads) that
    must reach the augmentation chain before ``.drop()`` removes it.
    """
    inference, summary, conditions, drop = _lists(cfg.adapter)
    keys = inference + summary + conditions + drop
    mask_key = getattr(cfg.adapter, "attention_mask_key", None)
    if mask_key:
        keys.append(str(mask_key))
    return keys


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
    mask_key = getattr(cfg, "attention_mask_key", None)
    if mask_key:
        adapter = adapter.rename(str(mask_key), "summary_attention_mask")
    if len(summary_variables) == 1:
        adapter = adapter.rename(summary_variables[0], "summary_variables")
    elif len(summary_variables) > 1:
        adapter = adapter.group(summary_variables, into="summary_variables")  # fusion
    if inference_conditions:
        adapter = adapter.concatenate(inference_conditions, into="inference_conditions")
    return adapter


def check_masked_backbone_occupancy(cfg) -> None:
    """Refuse the masked grid backbones with count-encoded occupancy under summary standardization.

    ``masked_time_series_transformer`` / ``masked_mlp`` read bin validity off the occupancy channels
    AFTER BayesFlow has standardized ``summary_variables``; only the sign-encoded
    ``summary_occupancy: valid`` survives that. With ``counts`` the network silently masks ~98 % of
    the bins (2026-09-22). Raising here is what keeps that from recurring.
    """
    if "summary_variables" not in list(cfg.training.get("standardize", [])):
        return
    sn = cfg.model.summary_network
    types = [str(sn.get("type", ""))]
    for spec in ((sn.get("params") or {}).get("backbones") or {}).values():
        types.append(str(spec.get("type", "")) if hasattr(spec, "get") else str(spec))
    if not any(t.startswith("masked_") and t != "masked_set_transformer" for t in types):
        return
    enc = str((cfg.augmentation.get("params") or {}).get("summary_occupancy", "counts")).lower()
    if enc != "valid":
        raise ValueError(
            "masked_time_series_transformer/masked_mlp need `augmentation.params.summary_occupancy=valid` "
            "when `training.standardize` includes summary_variables: the count encoding is z-scored "
            "before the network and its `count >= min_count` mask collapses (see stream_summary."
            "occupancy_encoding)."
        )
