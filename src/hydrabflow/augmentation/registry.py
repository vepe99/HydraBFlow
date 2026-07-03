"""Name -> augmentation-factory registry and builder.

An augmentation factory receives the shared ``params`` mapping from config **and a seeded NumPy
:class:`~numpy.random.Generator`**, and returns a callable ``batch -> batch``. Returning a closure
lets augmentations capture configuration (noise scale, target keys, ...) and their own private RNG
while staying compatible with BayesFlow's ``fit_offline(augmentations=[...])``.

Reproducibility contract: augmentations are *stochastic* (fresh draws every batch / epoch) yet
fully *reproducible* — all randomness flows from the single ``rng`` handed to
:func:`build_augmentations`, which is derived from ``cfg.seed`` in the train stage. Each step gets
its **own independent child generator** (via ``rng.spawn``), so a step's random stream does not
depend on which other steps are enabled or on their order. This mirrors the reference project's
``_split_key`` idiom (``jax.random.split``) but with NumPy generators.
"""

from __future__ import annotations

import numpy as np

from typing import Callable, Dict, List

# factory(params, rng) -> (batch -> batch)
Augmentation = Callable[[dict], dict]
_REGISTRY: Dict[str, Callable[..., Augmentation]] = {}


def register_augmentation(name: str):
    def _wrap(factory: Callable[..., Augmentation]) -> Callable[..., Augmentation]:
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"Augmentation '{name}' already registered")
        _REGISTRY[name] = factory
        return factory

    return _wrap


def build_augmentations(
    cfg, rng: np.random.Generator | None = None, context: dict | None = None
) -> List[Augmentation]:
    """Build the ordered augmentation list from ``cfg.augmentation`` (an ``AugmentationConfig``).

    ``rng`` seeds every augmentation's randomness. Pass the run's seeded generator (the train
    stage uses ``np.random.default_rng(cfg.seed)``) for reproducible-yet-stochastic behavior.
    If omitted, a default-seeded generator is used so the call still works in config-only contexts.

    ``context`` carries run-level objects an augmentation may need beyond its config params —
    notably the fitted :class:`~hydrabflow.preprocessing.base.PreprocessPipeline` under
    ``"pipeline"`` (e.g. per-stream standardization reads stats fitted on the train split).
    Factories that declare a third parameter receive it; two-parameter factories are unaffected.
    """
    import inspect

    from omegaconf import OmegaConf

    params = OmegaConf.to_container(cfg.params, resolve=True) if OmegaConf.is_config(cfg.params) else dict(cfg.params)
    if rng is None:
        rng = np.random.default_rng()

    steps = list(cfg.steps)
    # One independent, reproducible child stream per step (order-insensitive).
    child_rngs = rng.spawn(len(steps)) if steps else []

    augs: List[Augmentation] = []
    for name, child in zip(steps, child_rngs):
        if name not in _REGISTRY:
            raise KeyError(f"Unknown augmentation '{name}'. Registered: {sorted(_REGISTRY)}")
        factory = _REGISTRY[name]
        if len(inspect.signature(factory).parameters) >= 3:
            augs.append(factory(params, child, context or {}))
        else:
            augs.append(factory(params, child))
    return augs


def available_augmentations() -> list[str]:
    return sorted(_REGISTRY)
