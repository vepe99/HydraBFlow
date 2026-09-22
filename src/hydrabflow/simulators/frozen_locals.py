"""The guard both surrogate simulators need: locals baked into an artifact must stay pinned.

``stream_trihedron`` and ``stream_orbit_offset`` each replay a stored description of ONE stripping
history — the trihedron's frozen offsets, the orbit-offset template's correction / width / density
splines. The progenitor mass, its scale radius and the stripping age are inputs to that history,
not parameters the surrogate can respond to, so a config must pin them at the values the artifact
was built with.

Why this needs enforcing rather than documenting: ``AgamaStreamSimulator.local_parameter_names``
reads the FIRST stream's prior spec only, so freeing one of these silently promotes it to an
inference variable. The network would then be trained to infer a quantity the forward model
ignores, and the failure mode is a well-calibrated-looking posterior for a parameter that was never
used — invisible in every diagnostic this project runs.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

__all__ = ["FROZEN_LOCAL_KEYS", "check_frozen_locals"]

FROZEN_LOCAL_KEYS = ("m_progenitor", "a_progenitor", "t_end")


def check_frozen_locals(
    priors_local: Mapping,
    target_streams: Mapping[str, int],
    fiducial: Mapping[int, Mapping[str, float]],
    artifact: str,
) -> None:
    """Raise unless every frozen local is ``identity``-pinned at its ``fiducial`` value.

    ``fiducial`` maps stream index -> ``{key: value}``. All problems are collected into one error so
    a misconfigured run reports every mismatch at once rather than one per re-run.
    """
    problems = []
    for stream, j in target_streams.items():
        if j not in fiducial:
            problems.append(f"{stream}: {artifact} has no stream index {j}")
            continue
        for key in FROZEN_LOCAL_KEYS:
            spec = priors_local[stream][key]
            kind = str(spec["type"])
            pp = [float(v) for v in spec["prior_parameters"]]
            if kind != "identity":
                problems.append(
                    f"{stream}.{key}: prior type {kind!r}, but {artifact} freezes it "
                    f"(expected identity {fiducial[j][key]:g})"
                )
            elif not np.isclose(pp[0], fiducial[j][key], rtol=1e-6, atol=0.0):
                problems.append(
                    f"{stream}.{key}: pinned at {pp[0]:g}, {artifact} was built at "
                    f"{fiducial[j][key]:g}"
                )
    if problems:
        raise ValueError(
            f"{artifact} does not match the configured local priors:\n  " + "\n  ".join(problems)
        )
