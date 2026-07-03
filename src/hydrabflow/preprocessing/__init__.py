"""Deterministic, whole-dataset preprocessing.

This is the half of data handling that is NOT augmentation: transforms applied once to the full
dataset (NaN cleaning, train/val split, z-score standardization), fitted on the training split,
and saved to the run dir so evaluation / real-data inference replay the exact same transform.

Importing this package auto-imports every module in it, so any ``@register_step``-decorated
:class:`PreprocessStep` self-registers. To add your own step, just drop a new module in this
folder — no need to edit this file.
"""

from hydrabflow.preprocessing.base import PreprocessPipeline, PreprocessStep
from hydrabflow.preprocessing.registry import build_pipeline, register_step
from hydrabflow.utils.discovery import import_submodules

import_submodules(__name__, __path__)  # registers the shipped steps and any user steps

__all__ = [
    "PreprocessPipeline",
    "PreprocessStep",
    "build_pipeline",
    "register_step",
]
