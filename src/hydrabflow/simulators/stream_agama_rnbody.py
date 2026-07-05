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

_PLUMMER_MMAX = 0.99  # truncate the sampled enclosed-mass fraction (r <~ 12 scale radii)


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
) -> np.ndarray:
    """Restricted N-body stream: forward-integrate Plummer particles from the rewound orbit
    position, refitting the moving progenitor potential from the particles every ``tupd``."""
    # Rewind the progenitor's center to -T, then integrate it forward once, densely sampled:
    # the slices of this single trajectory drive the moving-potential center in every chunk.
    _, orbit_back = agama.orbit(
        potential=pot_host, ic=posvel_sat, time=-time_total, trajsize=2, accuracy=1e-10
    )
    xv_past = orbit_back[-1]
    n_knots = n_updates * traj_per_update + 1
    t_center, orbit_center = agama.orbit(
        potential=pot_host,
        ic=xv_past,
        time=time_total,
        timestart=-time_total,
        trajsize=n_knots,
        accuracy=1e-10,
    )

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
        )
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
        )
    except Exception:
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

        results = Parallel(n_jobs=self._n_workers)(
            delayed(_simulate_one_rnbody)(row, self._n_particles, self.obs_r_kpc, int(seed), opts)
            for row, seed in zip(rows, seeds)
        )
        xv = np.stack([r[0] for r in results], axis=0)  # (n, n_particles, 6)
        vcirc = np.stack([r[1] for r in results], axis=0)[..., None]  # (n, n_radii, 1)

        return {
            "sim_data_carthesian": xv,
            "sim_data_projected": sky_projection(xv),
            "vcirc_kms": vcirc,
        }
