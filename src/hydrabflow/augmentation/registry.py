"""Name -> augmentation factory, plus the builder.

A factory is ``(params, rng, context) -> (batch -> batch)``. Each step gets its own child generator
(``rng.spawn``), so its stream does not depend on which other steps are enabled or on their order:
augmentations are stochastic per batch yet reproducible from ``cfg.seed``. See ``noise.py``.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from hydrabflow.utils.registry import Registry

Augmentation = Callable[[dict], dict]

AUGMENTATIONS: Registry[Callable[..., Augmentation]] = Registry("augmentation")
register_augmentation = AUGMENTATIONS.add


def build_augmentations(
    cfg, rng: np.random.Generator | None = None, context: dict | None = None
) -> List[Augmentation]:
    """Build the ordered augmentation list from ``cfg.augmentation``."""
    from omegaconf import OmegaConf

    params = OmegaConf.to_container(cfg.params, resolve=True)
    steps = list(cfg.steps)
    rng = rng if rng is not None else np.random.default_rng()
    return [
        AUGMENTATIONS.get(name)(params, child, context or {})
        for name, child in zip(steps, rng.spawn(len(steps)))
    ]
