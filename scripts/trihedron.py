"""Compatibility shim: the Frenet-Serret remap now lives in the package.

It moved to ``hydrabflow.simulators.trihedron`` when ``stream_trihedron`` turned it from a
diagnostic into a forward model (its own docstring had always anticipated the move). The scripts
that grew up around it — ``compare_trihedron_vs_spray``, ``trihedron_prior_grid``,
``trihedron_fixed_mw22`` — keep importing ``trihedron`` unchanged through this module.

They all set ``HYDRABFLOW_NUM_GPUS=0`` before importing it, which is what keeps the package import
underneath from probing for a free GPU.
"""

from hydrabflow.simulators.trihedron import (  # noqa: F401
    Template,
    auto_window,
    build_template,
    nearest_time,
    orbit_window,
    remap,
    trihedron,
)

__all__ = [
    "Template",
    "auto_window",
    "build_template",
    "nearest_time",
    "orbit_window",
    "remap",
    "trihedron",
]
