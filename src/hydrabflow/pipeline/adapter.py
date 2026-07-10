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


def fill_stream_grid_from_simulator(cfg) -> None:
    """Align the training-time rotation-curve grid with the simulator's ``vcirc_kms`` grid.

    The rotation-curve observable can live on a non-default radial grid (e.g. the extended
    Zhou u Huang union grid, ``simulator.params.obs_r_grid: extended``, 50 radii). Two
    training-time components hardcode the default Zhou grid and would otherwise mismatch it:

      * the ``mask_vcirc_radii`` preprocessing step (raises if the observable's bin count differs
        from its configured ``radii``);
      * the ``add_noise_to_vcirc`` augmentation, whose per-bin sigma is drawn from the augmentation
        params' ``obs_sigma_vc`` / ``obs_r_kpc`` (defaulting to the Zhou grid).

    So when the simulator exposes a non-default grid, inject *its* radii + per-bin sigma into those
    two config nodes where the user has not set them explicitly. The simulator stays the single
    source of truth (``obs_r_kpc`` / ``obs_sigma_vc`` properties). No-op for simulators without a
    vcirc grid or when the config already pins the grid, so the default Zhou path is untouched.
    """
    from hydrabflow.simulators.registry import get_simulator

    try:
        simulator = get_simulator(cfg.simulator)
    except KeyError:
        return
    # Only stream simulators declaring a non-default grid need this; guard cheaply on the param.
    if str(getattr(getattr(cfg.simulator, "params", None), "obs_r_grid", "") or "") != "extended":
        return
    if not hasattr(simulator, "obs_r_kpc") or not hasattr(simulator, "obs_sigma_vc"):
        return

    from omegaconf import open_dict

    radii = [float(x) for x in simulator.obs_r_kpc]
    sigma = [float(x) for x in simulator.obs_sigma_vc]
    obs_vc = [float(x) for x in simulator.obs_vc_kms] if hasattr(simulator, "obs_vc_kms") else None

    # Preprocessing: give mask_vcirc_radii the full (untrimmed) grid so its bin-count check passes
    # and it trims to r >= r_min consistently with the augmentation; and give attach_observed_vcirc
    # (real-data eval) the observed curve on that same grid so its shape matches the mask.
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

    # Augmentation: give add_noise_to_vcirc the matching per-bin errors on the same grid.
    aug_params = getattr(getattr(cfg, "augmentation", None), "params", None)
    if aug_params is not None:
        with open_dict(aug_params):
            if not getattr(aug_params, "obs_r_kpc", None):
                aug_params.obs_r_kpc = radii
            if not getattr(aug_params, "obs_sigma_vc", None):
                aug_params.obs_sigma_vc = sigma


def adapter_keys(cfg) -> List[str]:
    """All dataset keys the adapter (``cfg.adapter``) consumes, in a stable order.

    ``drop`` is included so keys the adapter drops still survive :func:`select_adapter_keys`: a
    dropped key may be a *per-batch augmentation input* (e.g. the raw ``sim_data_projected`` star
    cloud a summary-statistics augmentation reads) that the network must NOT see — it has to reach
    the augmentation chain, then the adapter's ``.drop()`` removes it before the approximator.
    """
    keys = (
        _as_list(cfg.adapter.inference_variables)
        + _as_list(cfg.adapter.summary_variables)
        + _as_list(cfg.adapter.inference_conditions)
        + _as_list(cfg.adapter.drop)
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
