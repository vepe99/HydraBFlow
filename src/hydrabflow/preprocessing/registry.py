"""Name -> preprocessing step, plus the pipeline builder."""

from __future__ import annotations

from hydrabflow.preprocessing.base import PreprocessPipeline, PreprocessStep
from hydrabflow.utils.registry import Registry

STEPS: Registry[type[PreprocessStep]] = Registry("preprocessing step")
register_step = STEPS.add


def build_pipeline(cfg) -> PreprocessPipeline:
    """Build the pipeline from ``cfg.preprocessing``.

    Each entry in ``cfg.steps`` is a mapping ``{name: <registry key>, ...params}``; the remaining
    keys go to the registered step as keyword args.
    """
    from omegaconf import OmegaConf

    steps = []
    for entry in OmegaConf.to_container(cfg.steps, resolve=True):
        entry = dict(entry)
        steps.append(STEPS.get(entry.pop("name"))(**entry))
    return PreprocessPipeline(steps)
