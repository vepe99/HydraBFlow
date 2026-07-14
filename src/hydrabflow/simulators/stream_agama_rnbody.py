"""Restricted N-body stellar-stream forward model on AGAMA (CPU, joblib).

Same priors, sky projection and rotation curve as :mod:`stream_agama` (which it subclasses),
but the stream itself is generated with the **restricted N-body** method of agama's
``py/example_tidal_stream.py`` / ``py/tutorial_streams.ipynb`` instead of Fardal particle
spray: the progenitor is realized as ``n_particles`` self-consistently sampled Plummer
particles released at the orbit's rewound (past) position and integrated forward in the host
potential plus a *moving* progenitor potential that is periodically refit (monopole Multipole)
to the particles themselves. Tidal stripping and the progenitor's mass-loss history therefore
emerge dynamically instead of being imposed by a spray recipe.

Differences from the agama example, justified by the globular-cluster regime:

* no Chandrasekhar dynamical friction — for <~1e5 Msun progenitors it is negligible over the
  few-Gyr integration times used here (the example models a 1e9 Msun satellite);
* the progenitor's center follows the massless test-particle orbit (rewound then integrated
  forward once, densely sampled) rather than being re-estimated from the bound remnant — the
  self-friction of GC-mass systems is far below the phase-space prior widths.

Intended for recipe-robustness experiments: synthetic misspecification test sets and
spray-free training sets (see README "Future work"). Cost is one-to-two orders of magnitude
above particle spray, so keep ``n_workers`` modest and leave ``agama_num_threads: 1`` (each
worker pins its own agama OpenMP pool; total load ~= n_workers * agama_num_threads).
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from hydrabflow.simulators.registry import register_simulator
from hydrabflow.simulators.stream_agama import (
    AgamaStreamSimulator,
    _agama,
    _host_potential,
    _vcirc,
)
from hydrabflow.simulators.stream_common import sky_projection
from hydrabflow.utils.progress import joblib_row_progress
from hydrabflow.utils.quiet import quiet_worker

_PLUMMER_MMAX = 0.99  # truncate the sampled enclosed-mass fraction (r <~ 12 scale radii)


class _OrbitCapExceeded(Exception):
    """Raised when an ``agama.orbit`` call stopped at ``maxNumSteps`` instead of reaching the
    requested end time — i.e. a pathological orbit (e.g. near-radial/plunging) whose adaptive
    integrator step collapsed and would otherwise run for hours inside a single C call (a Python
    signal cannot interrupt it, since agama does not return to the interpreter mid-integration).
    The row is dropped to NaN in :func:`_simulate_one_rnbody`, exactly like any other failure."""


def _assert_reached(times, expected_final: float, span: float) -> None:
    """Raise :class:`_OrbitCapExceeded` unless every orbit reached ``expected_final``.

    A capped orbit stores its actual last-reached time, which falls short of the requested end.
    ``times`` is either a 1d timestamp array (single orbit) or an object array of such arrays (a
    bunch, ``result[:, 0]``). Tolerance is 1% of the integration ``span`` (reached times match the
    request exactly when uncapped, so this only ever fires on a genuine early stop)."""
    arr = np.atleast_1d(times)
    if arr.dtype == object:  # bunch: result[:, 0] is an object array of per-orbit time arrays
        reached = np.array([float(np.atleast_1d(t)[-1]) for t in arr])
    else:  # single orbit: a 1d array of timestamps
        reached = np.array([float(arr[-1])])
    if np.any(np.abs(reached - expected_final) > 0.01 * abs(span) + 1e-9):
        raise _OrbitCapExceeded()


def _plummer_sample(
    agama, n: int, mass: float, scale: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Positions/velocities (relative to the center) of an isotropic Plummer sphere.

    Positions by inverse-CDF of the enclosed-mass profile; velocity moduli by Aarseth's
    rejection sampling of q^2 (1-q^2)^(7/2) with v = q * v_esc(r); directions isotropic.
    """
    m = rng.uniform(0.0, _PLUMMER_MMAX, n)
    r = scale / np.sqrt(m ** (-2.0 / 3.0) - 1.0)

    def _isotropic(nn: int) -> np.ndarray:
        u = rng.uniform(-1.0, 1.0, nn)
        phi = rng.uniform(0.0, 2.0 * np.pi, nn)
        s = np.sqrt(1.0 - u * u)
        return np.column_stack([s * np.cos(phi), s * np.sin(phi), u])

    q = np.empty(n)
    filled = 0
    g_max = (2.0 / 9.0) * (7.0 / 9.0) ** 3.5  # peak of q^2 (1-q^2)^(7/2) at q = sqrt(2/9)
    while filled < n:
        cand = rng.uniform(0.0, 1.0, 2 * (n - filled))
        acc = cand[rng.uniform(0.0, g_max, cand.size) < cand**2 * (1.0 - cand**2) ** 3.5]
        take = min(acc.size, n - filled)
        q[filled : filled + take] = acc[:take]
        filled += take

    v_esc = np.sqrt(2.0 * agama.G * mass / np.sqrt(r**2 + scale**2))
    pos = _isotropic(n) * r[:, None]
    vel = _isotropic(n) * (q * v_esc)[:, None]
    return pos, vel


def _rnbody_stream(
    agama,
    pot_host,
    posvel_sat: np.ndarray,
    mass_sat: float,
    radius_sat: float,
    time_total: float,
    num_particles: int,
    rng: np.random.Generator,
    n_updates: int,
    traj_per_update: int,
    accuracy: float,
    max_num_steps: float,
    update_max_num_steps: float,
) -> np.ndarray:
    """Restricted N-body stream: forward-integrate Plummer particles from the rewound orbit
    position, refitting the moving progenitor potential from the particles every ``tupd``.

    Every ``agama.orbit`` call caps the integrator's step count and asserts it reached the
    requested end time; an orbit that hits the cap raises :class:`_OrbitCapExceeded` (-> NaN row).
    Two caps are used because the two kinds of orbit have very different legit step counts:
    ``max_num_steps`` for the progenitor's own rewind/center orbit (a single orbit integrated over
    the full time, ~1e3-1e4 legit steps), and the much tighter ``update_max_num_steps`` for the
    per-particle orbits inside each update (integrated over only ``tupd``, ~1e2 legit steps). The
    tight per-update cap bounds a moderately pathological row (whose 1000 particles are each slow
    but individually below 1e6 steps) to seconds instead of hours. The progenitor orbit is checked
    first, so a fully pathological row bails before the expensive update loop."""
    # Rewind the progenitor's center to -T, then integrate it forward once, densely sampled:
    # the slices of this single trajectory drive the moving-potential center in every chunk.
    t_back, orbit_back = agama.orbit(
        potential=pot_host, ic=posvel_sat, time=-time_total, trajsize=2,
        accuracy=1e-10, maxNumSteps=int(max_num_steps),
    )
    _assert_reached(t_back, -time_total, time_total)
    xv_past = orbit_back[-1]
    n_knots = n_updates * traj_per_update + 1
    t_center, orbit_center = agama.orbit(
        potential=pot_host,
        ic=xv_past,
        time=time_total,
        timestart=-time_total,
        trajsize=n_knots,
        accuracy=1e-10,
        maxNumSteps=int(max_num_steps),
    )
    _assert_reached(t_center, 0.0, time_total)

    pos, vel = _plummer_sample(agama, num_particles, mass_sat, radius_sat, rng)
    xv = np.tile(xv_past, (num_particles, 1))
    xv[:, 0:3] += pos
    xv[:, 3:6] += vel
    masses = np.full(num_particles, mass_sat / num_particles)

    pot_sat = agama.Potential(type="Plummer", mass=mass_sat, scaleRadius=radius_sat)
    tupd = time_total / n_updates
    for i in range(n_updates):
        knots = slice(i * traj_per_update, i * traj_per_update + traj_per_update + 1)
        pot_moving = agama.Potential(
            potential=pot_sat,
            center=np.column_stack([t_center[knots], orbit_center[knots]]),
        )
        result = agama.orbit(
            potential=agama.Potential(pot_host, pot_moving),
            ic=xv,
            time=tupd,
            timestart=t_center[knots][0],
            trajsize=1,
            accuracy=accuracy,
            maxNumSteps=int(update_max_num_steps),
        )
        _assert_reached(result[:, 0], t_center[knots][0] + tupd, tupd)
        xv = np.vstack(result[:, 1])
        # Refit the progenitor's own potential from its particles (monopole approximation,
        # as in the example: all particles contribute, stripped ones negligibly).
        center_now = orbit_center[knots][-1]
        try:
            pot_sat = agama.Potential(
                type="Multipole",
                particles=(xv[:, 0:3] - center_now[0:3], masses),
                symmetry="s",
            )
        except Exception:
            pass  # degenerate particle configuration: keep the previous satellite potential
    return xv


@quiet_worker
def _simulate_one_rnbody(
    p: Dict[str, float], n_particles: int, obs_r: np.ndarray, seed: int, opts: Dict[str, float]
):
    """joblib worker: one restricted-N-body stream + the rotation curve of its potential.

    Any failure yields NaN particles (the rotation curve is computed first and survives), so a
    single pathological row cannot abort a long dataset generation; downstream NaN cleaning
    drops such rows exactly like the spray simulator's invalid seeds.
    """
    agama = _agama()
    agama.setNumThreads(int(opts.get("agama_num_threads", 1)))
    rng = np.random.default_rng(seed)
    # getUnits()['time'] is a float [Myr] normally, but an astropy Quantity once astropy has
    # been imported in this process (agama >= 1.0.157) — take .value in that case.
    tu = agama.getUnits()["time"]
    time_unit_gyr = float(getattr(tu, "value", tu)) / 1e3

    pot_host = _host_potential(agama, p)
    vcirc = _vcirc(pot_host, obs_r)

    try:
        l0, b0, pml0, pmb0 = agama.transformCelestialCoords(
            agama.fromICRStoGalactic,
            p["ra"] * np.pi / 180,
            p["dec"] * np.pi / 180,
            p["mu_ra_cosdec"],
            p["mu_dec"],
        )
        posvel_sat = np.array(
            agama.getGalactocentricFromGalactic(l0, b0, p["r"], pml0 * 4.74, pmb0 * 4.74, p["vr"])
        )
        xv = _rnbody_stream(
            agama,
            pot_host,
            posvel_sat,
            mass_sat=p["m_progenitor"],
            radius_sat=p["a_progenitor"] / 1e3,  # pc -> kpc
            time_total=p["t_end"] / time_unit_gyr,
            num_particles=n_particles,
            rng=rng,
            n_updates=int(opts.get("n_updates", 12)),
            traj_per_update=int(opts.get("traj_per_update", 16)),
            accuracy=float(opts.get("accuracy", 1e-8)),
            max_num_steps=float(opts.get("max_num_steps", 1e6)),
            update_max_num_steps=float(opts.get("update_max_num_steps", 1e4)),
        )
    except Exception:  # _OrbitCapExceeded (pathological orbit hit the step cap) or agama failure
        xv = np.full((n_particles, 6), np.nan)
    return xv, vcirc


@register_simulator("stream_agama_rnbody")
class RestrictedNbodyStreamSimulator(AgamaStreamSimulator):
    """Stellar streams via restricted N-body (agama example_tidal_stream method)."""

    def _rnbody_opts(self) -> Dict[str, float]:
        return {
            "agama_num_threads": int(self.params.get("agama_num_threads", 1)),
            "n_updates": int(self.params.get("n_updates", 12)),
            "traj_per_update": int(self.params.get("traj_per_update", 16)),
            "accuracy": float(self.params.get("accuracy", 1e-8)),
            # Hard per-orbit step caps that guard against pathological orbits (near-radial/plunging
            # ones whose adaptive step collapses and would run for hours inside a single agama C
            # call — a Python timeout cannot interrupt that, since agama holds the GIL). A capped
            # orbit is detected (reached-time short) and its row dropped to NaN. Two caps because
            # the two orbit kinds differ ~50x in legit step count (measured):
            #   max_num_steps        - progenitor rewind/center orbit, full time, ~3e3 legit steps.
            #   update_max_num_steps - per-particle orbits over one update (tupd), ~1e2 legit steps;
            #     tight so a moderately-slow row (1000 particles each <1e6 steps) bails in ~150 s.
            "max_num_steps": float(self.params.get("max_num_steps", 1e6)),
            "update_max_num_steps": float(self.params.get("update_max_num_steps", 1e4)),
        }

    def simulate(
        self, params: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        from joblib import Parallel, delayed

        n = len(np.asarray(next(iter(params.values()))))
        rows = [
            {k: float(np.asarray(v).reshape(n, -1)[i, 0]) for k, v in params.items()}
            for i in range(n)
        ]
        seeds = rng.integers(0, 2**31 - 1, size=n)
        opts = self._rnbody_opts()

        # batch_size=1: restricted-N-body rows vary wildly in cost (fast rows vs the multi-minute
        # t_end=4 rows), so one row per dispatch keeps all n_workers busy instead of letting
        # joblib's 'auto' batching hand a few workers oversized batches while the rest idle.
        with joblib_row_progress():  # advance the run_chunked bar once per finished row
            results = Parallel(n_jobs=self._n_workers, batch_size=1)(
                delayed(_simulate_one_rnbody)(
                    row, self._n_particles, self.obs_r_kpc, int(seed), opts
                )
                for row, seed in zip(rows, seeds)
            )
        xv = np.stack([r[0] for r in results], axis=0)  # (n, n_particles, 6)
        vcirc = np.stack([r[1] for r in results], axis=0)[..., None]  # (n, n_radii, 1)

        return {
            "sim_data_carthesian": xv,
            "sim_data_projected": sky_projection(xv),
            "vcirc_kms": vcirc,
        }
