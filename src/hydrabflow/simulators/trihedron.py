"""Frenet-Serret stream remapping (Palau & Miralda-Escude 2023, MNRAS 524, 2124, Appendix D).

The idea: a stream simulation is expensive, but a stream is (to a good approximation) a fixed
internal structure carried along the progenitor's orbit. So simulate ONCE in a fiducial potential,
record every star relative to the moving Frenet-Serret trihedron of the progenitor orbit, and for
any new potential re-integrate only the progenitor orbit (milliseconds) and drop the stored
offsets back onto the new trihedra.

Per the paper, for each star ``e`` with present-day Galactocentric position ``x_e``:

    t_e     = argmin_t |x_o(t) - x_e|          over the orbit window t in [-T, +T]
    e1      = v / |v|                          at x_o(t_e)
    e2      = a / |a|,  a = d/dt (v / |v|)
    e3      = e1 x e2

and one stores ``t_e`` together with the star's position and velocity expressed in
``(e1, e2, e3)``. Two implementation notes:

* ``d/dt (v/|v|) = (g - (g.v^)v^) / |v|`` with ``g`` the acceleration, so ``e2`` is exactly the
  normalized *perpendicular* component of the gravitational acceleration. We take it analytically
  from ``potential.force()`` rather than by finite-differencing the orbit.
* The stored offsets are RELATIVE to the orbit point (``x_e - x_o(t_e)``, ``v_e - v_o(t_e)``),
  which is what "with respect to the reference frame defined by its trihedron" means and what
  keeps the reconstruction well behaved when the new orbit has a different speed. ``velocity=
  "absolute"`` stores ``v_e`` itself instead, since the paper's wording is not unambiguous.

The module is deliberately free of hydrabflow imports (numpy + a live ``agama`` module handle
only), so it can be lifted into ``src/hydrabflow/simulators/`` as a worker if the approximation
turns out to be good enough to generate datasets with.

Times are in agama's internal time unit throughout (~0.978 Gyr under the (kpc, km/s, Msun) unit
system this project uses); ``t = 0`` is the present day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

_STAR_BLOCK = 512  # stars per block in the nearest-orbit-point search (bounds peak memory)


@dataclass
class Template:
    """A stream frozen into the trihedron frame of its progenitor's orbit."""

    t_hat: np.ndarray  # (P,) time of the nearest orbit point, agama time units
    c: np.ndarray  # (P,3) position offset in the (e1,e2,e3) basis [kpc]
    w: np.ndarray  # (P,3) velocity (relative or absolute) in the basis [km/s]
    t_knots: np.ndarray  # (K,) orbit sampling grid, must be reused by remap()
    velocity: str = "relative"
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def n_stars(self) -> int:
        return int(self.t_hat.shape[0])


# --------------------------------------------------------------------------------------------
# orbit window + trihedron
# --------------------------------------------------------------------------------------------


def orbit_window(agama, pot, posvel_sat: np.ndarray, T: float, n_knots: int):
    """Progenitor orbit densely sampled on ``[-T, +T]`` about the present day.

    Rewind to ``-T`` first and then integrate forward over ``2T`` in one call, so the samples land
    on a regular grid (agama spaces them as ``dt = time/(trajsize-1)``). Same idiom as
    ``stream_agama_rnbody._rnbody_stream``'s centre trajectory.
    """
    _, back = agama.orbit(
        potential=pot, ic=posvel_sat, time=-T, trajsize=2, accuracy=1e-10, dtype=float
    )
    xv_past = np.asarray(back[-1], dtype=float)
    t, orb = agama.orbit(
        potential=pot,
        ic=xv_past,
        time=2.0 * T,
        timestart=-T,
        trajsize=int(n_knots),
        accuracy=1e-10,
        dtype=float,
    )
    t = np.asarray(t, dtype=float)
    orb = np.asarray(orb, dtype=float)
    if not np.isfinite(orb).all() or abs(t[-1] - T) > 1e-6 * max(T, 1.0):
        raise RuntimeError(f"progenitor orbit did not reach +T ({t[-1]:.4g} vs {T:.4g})")
    return t, orb


def trihedron(pot, x: np.ndarray, v: np.ndarray):
    """``(e1, e2, e3)`` at Galactocentric states ``(x, v)``, each ``(N,3)``.

    ``e2`` is the normalized perpendicular acceleration, which is the direction of the paper's
    ``a = d/dt(v/|v|)`` exactly (they differ only by the positive factor ``1/|v|``). Where the
    perpendicular acceleration vanishes (a locally straight orbit) the frame is degenerate; we fall
    back to an arbitrary vector orthogonal to ``e1`` and count the occurrence.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    v = np.atleast_2d(np.asarray(v, dtype=float))
    e1 = v / np.linalg.norm(v, axis=1, keepdims=True)
    g = np.atleast_2d(np.asarray(pot.force(x), dtype=float))
    g_perp = g - np.sum(g * e1, axis=1, keepdims=True) * e1
    norm = np.linalg.norm(g_perp, axis=1, keepdims=True)
    degenerate = (norm[:, 0] <= 1e-12 * np.linalg.norm(g, axis=1)).astype(bool)
    if degenerate.any():  # any fixed vector not parallel to e1 will do
        alt = np.zeros_like(e1)
        alt[np.arange(len(e1)), np.argmin(np.abs(e1), axis=1)] = 1.0
        alt = alt - np.sum(alt * e1, axis=1, keepdims=True) * e1
        g_perp = np.where(degenerate[:, None], alt, g_perp)
        norm = np.linalg.norm(g_perp, axis=1, keepdims=True)
    e2 = g_perp / norm
    e3 = np.cross(e1, e2)
    return e1, e2, e3, int(degenerate.sum())


def _hermite(t_knots: np.ndarray, orb: np.ndarray, g_knots: np.ndarray, t: np.ndarray):
    """Cubic Hermite interpolation of the orbit at arbitrary times ``t``.

    Positions use ``(x, v)`` at the bracketing knots and velocities use ``(v, g)`` — the derivative
    of each quantity is known exactly, so no spline fitting (and no scipy) is needed and the error
    is O(dt^4). Returns ``(x, v)``.
    """
    h = float(t_knots[1] - t_knots[0])
    k = np.clip(np.searchsorted(t_knots, t, side="right") - 1, 0, len(t_knots) - 2)
    s = (t - t_knots[k]) / h
    s2, s3 = s * s, s * s * s
    h00 = (2 * s3 - 3 * s2 + 1)[:, None]
    h10 = (s3 - 2 * s2 + s)[:, None] * h
    h01 = (-2 * s3 + 3 * s2)[:, None]
    h11 = (s3 - s2)[:, None] * h
    x0, v0 = orb[k, 0:3], orb[k, 3:6]
    x1, v1 = orb[k + 1, 0:3], orb[k + 1, 3:6]
    g0, g1 = g_knots[k], g_knots[k + 1]
    return (h00 * x0 + h10 * v0 + h01 * x1 + h11 * v1, h00 * v0 + h10 * g0 + h01 * v1 + h11 * g1)


def _state_at(pot, t_knots: np.ndarray, orb: np.ndarray, t: np.ndarray):
    """Orbit state and trihedron at arbitrary times ``t`` on the knot grid."""
    g_knots = np.atleast_2d(np.asarray(pot.force(orb[:, 0:3]), dtype=float))
    x, v = _hermite(t_knots, orb, g_knots, t)
    e1, e2, e3, n_degen = trihedron(pot, x, v)
    return x, v, np.stack([e1, e2, e3], axis=1), n_degen  # basis (N,3,3), rows = e1,e2,e3


# --------------------------------------------------------------------------------------------
# nearest orbit point
# --------------------------------------------------------------------------------------------


def nearest_time(t_knots: np.ndarray, orb: np.ndarray, x_star: np.ndarray):
    """``t_hat`` per star plus the two diagnostics that decide whether the labels mean anything.

    ``boundary_frac``  fraction of stars whose nearest knot is an endpoint of the window: their
                       true nearest point lies outside ``[-T, +T]``, so ``T`` is too small.
    ``ambiguous_frac`` fraction of stars whose distance-to-orbit has more than one local minimum
                       within 2x the global minimum distance — the argmin has jumped between
                       orbital wraps and the star's label is not unique. Reported, not repaired.
    """
    xo = orb[:, 0:3]
    n_star, n_knot = len(x_star), len(t_knots)
    t_hat = np.empty(n_star)
    n_boundary = n_ambiguous = 0
    for lo in range(0, n_star, _STAR_BLOCK):
        block = x_star[lo : lo + _STAR_BLOCK]
        d2 = np.sum((xo[None, :, :] - block[:, None, :]) ** 2, axis=2)  # (b, K)
        k = np.argmin(d2, axis=1)
        rows = np.arange(len(block))
        n_boundary += int(np.sum((k == 0) | (k == n_knot - 1)))
        # local minima of d2 along the knot axis, counted only where they are competitive
        interior = d2[:, 1:-1]
        is_min = (interior < d2[:, :-2]) & (interior < d2[:, 2:])
        competitive = interior < 4.0 * d2[rows, k][:, None]
        n_ambiguous += int(np.sum(np.sum(is_min & competitive, axis=1) > 1))
        # parabolic refinement on the three knots bracketing the minimum
        ki = np.clip(k, 1, n_knot - 2)
        y0, y1, y2 = d2[rows, ki - 1], d2[rows, ki], d2[rows, ki + 1]
        denom = y0 - 2 * y1 + y2
        shift = np.where(np.abs(denom) > 0, 0.5 * (y0 - y2) / np.where(denom == 0, 1.0, denom), 0.0)
        shift = np.clip(shift, -1.0, 1.0)
        h = t_knots[1] - t_knots[0]
        t_hat[lo : lo + len(block)] = np.where(k == ki, t_knots[ki] + shift * h, t_knots[k])
    t_hat = np.clip(t_hat, t_knots[0], t_knots[-1])
    return t_hat, dict(
        boundary_frac=n_boundary / max(n_star, 1),
        ambiguous_frac=n_ambiguous / max(n_star, 1),
    )


# --------------------------------------------------------------------------------------------
# build / remap
# --------------------------------------------------------------------------------------------


def build_template(
    agama,
    pot,
    posvel_sat: np.ndarray,
    xv_stars: np.ndarray,
    T: float,
    n_knots: int,
    velocity: str = "relative",
) -> Template:
    """Freeze a simulated stream into the trihedron frame of its progenitor's orbit.

    ``xv_stars`` is ``(P,6)`` present-day Galactocentric phase space (what ``_rnbody_stream``
    returns). NaN stars (failed orbits) are carried through as NaN.
    """
    if velocity not in ("relative", "absolute"):
        raise ValueError(f"velocity must be 'relative' or 'absolute', got {velocity!r}")
    xv_stars = np.asarray(xv_stars, dtype=float)
    finite = np.isfinite(xv_stars).all(axis=1)
    t_knots, orb = orbit_window(agama, pot, posvel_sat, T, n_knots)

    t_hat = np.full(len(xv_stars), np.nan)
    t_hat[finite], diag = nearest_time(t_knots, orb, xv_stars[finite, 0:3])

    c = np.full((len(xv_stars), 3), np.nan)
    w = np.full((len(xv_stars), 3), np.nan)
    if finite.any():
        xo, vo, basis, n_degen = _state_at(pot, t_knots, orb, t_hat[finite])
        dx = xv_stars[finite, 0:3] - xo
        dv = xv_stars[finite, 3:6] - (vo if velocity == "relative" else 0.0)
        c[finite] = np.einsum("nk,njk->nj", dx, basis)
        w[finite] = np.einsum("nk,njk->nj", dv, basis)
        diag["degenerate_frame_frac"] = n_degen / max(int(finite.sum()), 1)
    diag["nan_star_frac"] = float(1.0 - finite.mean())
    diag["T"] = float(T)
    diag["n_knots"] = int(n_knots)
    return Template(t_hat=t_hat, c=c, w=w, t_knots=t_knots, velocity=velocity, diagnostics=diag)


def remap(agama, pot_new, posvel_sat: np.ndarray, tpl: Template) -> np.ndarray:
    """Reconstruct the stream in a new potential. Returns ``(P,6)`` Galactocentric phase space.

    Only the progenitor orbit is re-integrated; the stars are placed at their stored offsets in the
    new trihedra. ``posvel_sat`` is the progenitor's present-day 6D state, which is an observable
    and so is normally unchanged between the fiducial and the new potential.
    """
    T = float(tpl.t_knots[-1])
    t_knots, orb = orbit_window(agama, pot_new, posvel_sat, T, len(tpl.t_knots))
    finite = np.isfinite(tpl.t_hat)
    out = np.full((tpl.n_stars, 6), np.nan)
    if not finite.any():
        return out
    xo, vo, basis, _ = _state_at(pot_new, t_knots, orb, tpl.t_hat[finite])
    out[finite, 0:3] = xo + np.einsum("nj,njk->nk", tpl.c[finite], basis)
    out[finite, 3:6] = np.einsum("nj,njk->nk", tpl.w[finite], basis)
    if tpl.velocity == "relative":
        out[finite, 3:6] += vo
    return out


def auto_window(
    agama,
    pot,
    posvel_sat: np.ndarray,
    xv_stars: np.ndarray,
    T0: float,
    n_knots: int,
    t_max: float,
    tol: float = 1e-3,
    growth: float = 1.5,
    max_tries: int = 8,
    velocity: str = "relative",
):
    """Grow ``T`` until fewer than ``tol`` of stars pin to the window boundary.

    The paper used T = 40-60 Myr for 1.5-4 Gyr streams; our arms span ~100 deg, so the orbit
    segment the stream actually covers can be far longer and a too-small window silently piles
    stars onto the endpoints. Knot spacing is held roughly fixed as ``T`` grows.
    """
    T, knots = float(T0), int(n_knots)
    for _ in range(max_tries):
        tpl = build_template(agama, pot, posvel_sat, xv_stars, T, knots, velocity=velocity)
        if tpl.diagnostics["boundary_frac"] < tol or T >= t_max:
            return tpl, T, knots
        T = min(T * growth, t_max)
        knots = int(round(n_knots * T / T0))
    return tpl, T, knots
