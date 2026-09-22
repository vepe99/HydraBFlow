"""Forward model: progenitor orbit + stored DeltaTheta(phi1) corrections (Ibata et al. 2024).

Each row re-integrates only the progenitor's orbit in that row's potential, projects it into
observation space, and paints the stream on with the template's degree-4 correction and dispersion
splines -- a few milliseconds per stream against ~10-30 s for the restricted N-body at 10^4
particles. The physics and the fitting live in :mod:`hydrabflow.simulators.orbit_offset`; this
module is only the pipeline wrapper.

FREE: every global potential parameter (they change the orbit, which is what gets re-integrated)
and the four present-day phase-space coordinates ``vr``, ``r``, ``mu_ra_cosdec``, ``mu_dec``.

PINNED: ``m_progenitor``, ``a_progenitor`` and ``t_end``. They set the stripping history, which the
template encodes as a fixed correction, width and along-track density -- the model cannot respond to
them, so :func:`~hydrabflow.simulators.frozen_locals.check_frozen_locals` raises if a config frees
one. See that module for why this is enforced rather than documented.

What the model cannot represent, by construction:

* **the along-track density**, which is frozen at the fiducial. The potential moves the orbit, not
  the stars along it. Same structural limit as ``stream_trihedron``'s frozen ``t_hat``.
* **progenitor survival**: there is no progenitor and no stripping, so no ``m_bound_final``.
* **correlations between observables**, since the dispersions are sampled independently per
  observable. Measured cost at the fiducial: the phi2 width comes out 0.85 / 0.91 / 1.03 of the
  simulation's for Pal5 / NGC3201 / M68 (the position residual's ra-dec correlation is what phi2
  is made of). Every mean track is reproduced to <= 0.045 deg.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
from scipy.interpolate import BSpline

from ..registry import register_simulator
from ..utils.quiet import quiet_worker
from . import orbit_offset as _oo
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

# Templates are small but every joblib worker needs all of them, so load once per process; loky
# reuses workers across the rows of a chunk, so this is paid once each.
_TEMPLATE_CACHE: Dict[str, Dict[int, _oo.OffsetTemplate]] = {}


def load_templates(path: str) -> Dict[int, _oo.OffsetTemplate]:
    """Per-stream :class:`OffsetTemplate`s from a ``build_orbit_offset_template.py`` npz."""
    if path not in _TEMPLATE_CACHE:
        with np.load(path, allow_pickle=False) as f:
            k = int(f["degree"])
            T0, knot_dt = float(f["T0"]), float(f["knot_dt"])

            def spline(kind: str, obs: str, j: int) -> BSpline:
                return BSpline(f[f"{kind}_{obs}_t_{j}"], f[f"{kind}_{obs}_c_{j}"], k)

            _TEMPLATE_CACHE[path] = {
                int(j): _oo.OffsetTemplate(
                    R=f[f"R_{int(j)}"],
                    T0=T0,
                    knot_dt=knot_dt,
                    delta={o: spline("delta", o, int(j)) for o in _oo.OBS_NAMES},
                    sigma={o: spline("sigma", o, int(j)) for o in _oo.OBS_NAMES},
                    sigma_bounds={
                        o: tuple(float(v) for v in f[f"sigma_bounds_{o}_{int(j)}"])
                        for o in _oo.OBS_NAMES
                    },
                    phi1_cdf_q=f[f"phi1_cdf_q_{int(j)}"],
                    phi1_cdf_x=f[f"phi1_cdf_x_{int(j)}"],
                    phi1_lo=float(f[f"phi1_lo_{int(j)}"]),
                    phi1_hi=float(f[f"phi1_hi_{int(j)}"]),
                )
                for j in f["streams"]
            }
    return _TEMPLATE_CACHE[path]


def template_fiducial_locals(path: str) -> Dict[int, Dict[str, float]]:
    """The stripping history each stream's template was built at."""
    with np.load(path, allow_pickle=False) as f:
        return {
            int(j): {key: float(f[f"fiducial_{key}_{int(j)}"]) for key in FROZEN_LOCAL_KEYS}
            for j in f["streams"]
        }


@quiet_worker
def _simulate_one_orbit_offset(
    p: Dict[str, float],
    template_path: str,
    n_particles: int,
    obs_r: np.ndarray,
    seed: int,
    pot_cfg: Mapping | None = None,
    ancillary: Mapping | None = None,
    max_miss_frac: float = 0.05,
):
    agama = _agama()
    # Potential-only outputs first, so a failed orbit still leaves a usable rotation curve.
    pot_host = _host_potential(agama, p, pot_cfg)
    frame = _solar_frame(agama, pot_host, p)
    vcirc = _vcirc(pot_host, obs_r)
    anc = _ancillary_observables(agama, pot_host, p, pot_cfg, ancillary, r0=frame[0])
    halo_derived = _m200c_derived(agama, p, pot_cfg)

    tpl = load_templates(template_path)[int(round(p["j"]))]
    try:
        # Verbatim stream_agama._simulate_one: the observed ICRS coordinates are fixed, but the
        # frame they are converted through depends on the potential.
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
        track = _oo.orbit_track_covering(
            agama, pot_host, posvel_sat, frame, tpl.R,
            tpl.phi1_lo, tpl.phi1_hi, tpl.T0, tpl.knot_dt,
        )
        obs, _, miss = _oo.sample_stream(
            tpl, track, int(n_particles), np.random.default_rng(int(seed))
        )
        if miss > float(max_miss_frac):
            raise RuntimeError(f"orbit reaches only {1 - miss:.3f} of the stream in phi1")
        xv = _oo.observables_to_cartesian(agama, obs, frame)
        if not np.isfinite(xv).all():
            raise RuntimeError("non-finite Galactocentric state")
    except Exception:
        xv = np.full((int(n_particles), 6), np.nan)
    return xv, vcirc, anc, halo_derived, None, frame


@register_simulator("stream_orbit_offset")
class OrbitOffsetStreamSimulator(AgamaStreamSimulator):
    """Streams painted onto the progenitor orbit from stored spline corrections."""

    @property
    def _template_path(self) -> str:
        path = self.params.get("orbit_offset_template")
        if not path:
            raise KeyError(
                "stream_orbit_offset requires params.orbit_offset_template (build it with "
                "scripts/build_orbit_offset_template.py)"
            )
        return str(path)

    def _check_template_matches_priors(self) -> None:
        check_frozen_locals(
            self._priors_local,
            self.target_streams,
            template_fiducial_locals(self._template_path),
            "orbit-offset template",
        )

    def _row_jobs(self, rows, seeds):
        """Swap in the orbit+offset worker; the base class keeps the dispatch and the whole output
        assembly (rotation curve, ancillary observables, derived m200_c diagnostics, the in-window
        storage cap)."""
        from joblib import delayed

        self._check_template_matches_priors()
        path, pot_cfg, ancillary = self._template_path, self._pot_cfg, self._ancillary_spec
        max_miss = float(self.params.get("max_phi1_miss_frac", 0.05))
        n_particles = self._n_particles
        return [
            delayed(_simulate_one_orbit_offset)(
                row, path, n_particles, self.obs_r_kpc, int(seed), pot_cfg, ancillary, max_miss
            )
            for row, seed in zip(rows, seeds)
        ]
