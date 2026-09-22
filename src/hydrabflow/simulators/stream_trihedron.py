"""Stellar streams by Frenet-Serret remapping of ONE restricted-N-body simulation.

Palau & Miralda-Escude (2023, MNRAS 524, 2124) Appendix D: a stream is, to a good approximation, a
fixed internal structure carried along the progenitor's orbit. So simulate once in a fiducial
potential, freeze every star into the moving trihedron of the progenitor orbit
(``hydrabflow.simulators.trihedron``), and for any new potential re-integrate only the progenitor
orbit (milliseconds) and drop the stored offsets onto the new trihedra. Measured here: ~40 ms per
stream against ~10-30 s for the restricted N-body at 10^4 particles.

What varies and what cannot
---------------------------
The remap re-integrates the progenitor orbit, so anything that changes that orbit is free: every
global potential parameter, and the progenitor's present-day phase space (``vr``, ``r``,
``mu_ra_cosdec``, ``mu_dec``) — the conversion from observed ICRS coordinates to a Galactocentric
state is byte-identical to ``stream_agama._simulate_one``, so those locals move the orbit exactly
as they do in the real simulator.

What CANNOT vary is the stripping history: ``m_progenitor``, ``a_progenitor`` and ``t_end`` are
baked into the stored offsets. A config using this simulator must pin them (``identity``) at the
fiducial run's values; ``_check_template_matches_priors`` raises at construction otherwise, because
a silently-ignored prior would put a parameter in ``local_parameter_names`` that the forward model
does not respond to.

Accuracy envelope (measured, sessions 2026-08-28)
------------------------------------------------
``t_hat`` is frozen, so the remap cannot redistribute stars ALONG the track. A halo-flattening
change bends the orbit sideways, which the stored perpendicular offsets do capture; a halo-MASS
change is mostly an orbital-period rescaling, which they do not. Against restricted-N-body truth at
10^6 particles the remap beat particle spray across a flattening sweep for M68 but lost on the
oblate side for NGC3201 and at both extremes for Pal5, and lost on the mass axis for every stream.
It is exact at its own fiducial potential and degrades away from it, so it is a surrogate with a
usable window around the template, not a drop-in replacement over an arbitrary prior. The remap
also inherits the fiducial's remnant, so it cannot represent potential-dependent progenitor
survival and reports no ``m_bound_final``.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from ..registry import register_simulator
from ..utils.quiet import quiet_worker
from . import trihedron as _tri
from .frozen_locals import FROZEN_LOCAL_KEYS, check_frozen_locals
from .stream_agama import (
    AgamaStreamSimulator,
    _agama,
    _ancillary_observables,
    _host_potential,
    _m200c_derived,
    _solar_frame,
    _vcirc,
)

# Templates are a few MB per stream and every joblib worker needs all of them, so load once per
# process. joblib's loky workers are reused across the rows of a chunk, so this is paid once each.
_TEMPLATE_CACHE: Dict[str, Dict[int, "_tri.Template"]] = {}


def load_templates(path: str) -> Dict[int, "_tri.Template"]:
    """Per-stream ``Template``s from a ``build_trihedron_template.py`` npz, cached per process."""
    if path not in _TEMPLATE_CACHE:
        with np.load(path, allow_pickle=False) as f:
            t_knots = f["t_knots"]
            velocity = str(f["velocity"])
            js = [int(j) for j in f["streams"]]
            _TEMPLATE_CACHE[path] = {
                j: _tri.Template(
                    t_hat=f[f"t_hat_{j}"],
                    c=f[f"c_{j}"],
                    w=f[f"w_{j}"],
                    t_knots=t_knots,
                    velocity=velocity,
                )
                for j in js
            }
    return _TEMPLATE_CACHE[path]


def template_fiducial_locals(path: str) -> Dict[int, Dict[str, float]]:
    """The frozen locals (``m_progenitor``/``a_progenitor``/``t_end``) the template was built from."""
    with np.load(path, allow_pickle=False) as f:
        return {
            int(j): {k: float(f[f"fiducial_{k}_{int(j)}"]) for k in FROZEN_LOCAL_KEYS}
            for j in f["streams"]
        }


@quiet_worker
def _simulate_one_trihedron(
    p: Dict[str, float], template_path: str, obs_r: np.ndarray, seed: int,
    pot_cfg: Mapping | None = None, ancillary: Mapping | None = None,
):
    """joblib worker: one remapped stream + the rotation curve of its potential (+ the optional
    Ibata ancillary observables), returning ``stream_agama._row_jobs``'s 6-tuple contract.

    ``seed`` is unused — the remap is deterministic given the row (the randomness lived in the one
    fiducial N-body run) — but it stays in the signature so the worker is a drop-in for the other
    two and the caller needs no special case.

    As in the restricted-N-body worker, the potential-only outputs are computed before the stream so
    a failed orbit still yields a usable rotation curve and the row is dropped downstream by its
    NaN particles rather than aborting the run.
    """
    agama = _agama()
    pot_host = _host_potential(agama, p, pot_cfg)
    frame = _solar_frame(agama, pot_host, p)
    vcirc = _vcirc(pot_host, obs_r)
    anc = _ancillary_observables(agama, pot_host, p, pot_cfg, ancillary, r0=frame[0])
    halo_derived = _m200c_derived(agama, p, pot_cfg)

    tpl = load_templates(template_path)[int(round(p["j"]))]
    try:
        # Verbatim stream_agama._simulate_one: the observed ICRS coordinates are fixed inputs, but
        # the frame they are converted through depends on this row's potential and solar draw.
        l0, b0, pml0, pmb0 = agama.transformCelestialCoords(
            agama.fromICRStoGalactic,
            p["ra"] * np.pi / 180,
            p["dec"] * np.pi / 180,
            p["mu_ra_cosdec"],
            p["mu_dec"],
        )
        posvel_sat = np.array(
            agama.getGalactocentricFromGalactic(
                l0, b0, p["r"], pml0 * 4.74, pmb0 * 4.74, p["vr"],
                galcen_distance=frame[0], galcen_v_sun=frame[1:4], z_sun=frame[4],
            )
        )
        xv = _tri.remap(agama, pot_host, posvel_sat, tpl)
    except Exception:  # a pathological orbit that agama cannot integrate over the template window
        xv = np.full((tpl.n_stars, 6), np.nan)
    # m_bound_final is deliberately None: the remap replays the fiducial's already-stripped
    # population, so any bound mass it reported would be the fiducial's input dressed up as a
    # measurement (the same reason the spray worker refuses to report one).
    return xv, vcirc, anc, halo_derived, None, frame


@register_simulator("stream_trihedron")
class TrihedronStreamSimulator(AgamaStreamSimulator):
    """Streams remapped from a stored trihedron template (see the module docstring)."""

    @property
    def _template_path(self) -> str:
        path = self.params.get("trihedron_template")
        if not path:
            raise KeyError(
                "stream_trihedron requires params.trihedron_template (build it with "
                "scripts/build_trihedron_template.py)"
            )
        return str(path)

    @property
    def _n_particles(self) -> int:
        """The template's star count, NOT ``params.n_particles``: the remap replays exactly the
        stars the fiducial simulation produced."""
        tpl = load_templates(self._template_path)
        return int(next(iter(tpl.values())).n_stars)

    def _check_template_matches_priors(self) -> None:
        """Every frozen local must be pinned at the value the template was built with."""
        check_frozen_locals(
            self._priors_local,
            self.target_streams,
            template_fiducial_locals(self._template_path),
            "trihedron template",
        )

    def _row_jobs(self, rows, seeds):
        """Swap the forward-model worker for the remap; the base class's ``simulate`` keeps the
        dispatch and the whole output assembly (rotation curve, ancillary observables, derived
        m200_c diagnostics, the in-window storage cap)."""
        from joblib import delayed

        self._check_template_matches_priors()
        path, pot_cfg, ancillary = self._template_path, self._pot_cfg, self._ancillary_spec
        return [
            delayed(_simulate_one_trihedron)(
                row, path, self.obs_r_kpc, int(seed), pot_cfg, ancillary
            )
            for row, seed in zip(rows, seeds)
        ]
