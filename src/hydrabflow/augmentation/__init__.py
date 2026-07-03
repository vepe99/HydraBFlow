"""Per-batch, stochastic augmentations applied inside ``workflow.fit_offline``.

Each augmentation is a callable ``batch -> batch`` (a dict of arrays). They are resolved by name
from the registry and composed in config order. This is the counterpart to the preprocessing
module: augmentations are random and re-drawn every epoch, preprocessing is deterministic and
applied once.

Importing this package auto-imports every module in it, so any ``@register_augmentation``-
decorated factory self-registers. To add your own augmentation, just drop a new module in this
folder — no need to edit this file.
"""

from hydrabflow.augmentation.registry import build_augmentations, register_augmentation
from hydrabflow.utils.discovery import import_submodules

import_submodules(__name__, __path__)  # registers the shipped examples and any user augmentations

__all__ = ["build_augmentations", "register_augmentation"]
