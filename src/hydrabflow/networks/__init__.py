"""BayesFlow network builders, resolved by ``cfg.type`` through registries.

Importing this package auto-imports every module in it, so a dropped module with an
``@register_summary_network`` / ``@register_inference_network`` decorated builder self-registers.
To add a custom architecture, just drop a new module in this folder — no need to edit this file.
"""

from hydrabflow.networks.factory import (
    build_inference_network,
    build_summary_network,
    register_inference_network,
    register_summary_network,
)
from hydrabflow.utils.discovery import import_submodules

import_submodules(__name__, __path__)  # registers any user architectures

__all__ = [
    "build_summary_network",
    "build_inference_network",
    "register_summary_network",
    "register_inference_network",
]
