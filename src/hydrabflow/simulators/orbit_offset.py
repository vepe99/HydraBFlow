"""Streams as a progenitor orbit plus smooth corrections DeltaTheta(phi1) -- Ibata et al. 2024.

The method is Section 4 of "Charting the Galactic Acceleration Field II" (ApJ 967, 89;
arXiv:2311.17202), the same paper this project takes the STREAMFINDER frames and the 30 pc
progenitor radius from:

    "stellar streams do not precisely delineate the orbital path of their progenitors. To overcome
    this complication we will proceed in an iterative manner, to find plausible functions
    DeltaTheta(phi1) in the derived mass model that correct the offset between the stream and the
    progenitor orbit."

    "the DeltaTheta(phi1) is then calculated independently for each observable Theta as a fourth
    order polynomial fit to the best fit stream minus the corresponding progenitor orbit."

So a stream is generated in three moves: integrate the progenitor orbit in the candidate potential,
project it into observation space, and at each phi1 displace by the stored correction. The six
corrected observables are Ibata's own list -- (ra, dec, distance modulus, mu_phi1, mu_phi2, v_los).

Here the correction is a **degree-4 spline** rather than a plain quartic. That is a strict
generalization: a degree-4 spline with zero interior knots IS a quartic polynomial (see
``fit_offset_spline`` and the test that pins it against ``numpy.polyfit``), and interior knots let
the correction bend where one quartic cannot.

Two additions the paper does not need, because a generative simulator needs more than a mean track:

* a second degree-4 spline per observable for the **dispersion** sigma_Theta(phi1), fitted in
  log sigma so the evaluated width is positive everywhere. Ibata models the width as a fixed 50 pc
  Gaussian, which is far thinner than these globular-cluster streams actually are, and this
  project's diagnostics are built on per-phi1-bin dispersions -- so the width is fitted, not assumed.
* an empirical **phi1 density**, stored as a quantile function and sampled by inverse CDF.

What the model cannot do, by construction: the along-track density is frozen. The potential moves
the orbit; it does not move stars along it. This is the same structural limit as the trihedron
surrogate's frozen ``t_hat``, and it is why ``scripts/validate_surrogates_vs_nbody.py`` exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

from . import stream_frame as _sf
from .stream_common import sky_projection
from .trihedron import orbit_window

__all__ = [
    "OBS_NAMES",
    "SPLINE_DEGREE",
    "OffsetTemplate",
    "OrbitTrack",
    "build_offset_template",
    "dm_from_distance",
    "distance_from_dm",
    "eval_spline",
    "fit_offset_spline",
    "observables_to_cartesian",
    "orbit_track",
    "sample_stream",
]

#: The observables Ibata corrects, in the order they are stored everywhere in this module.
OBS_NAMES = ("ra", "dec", "dm", "mu_phi1", "mu_phi2", "vlos")

#: "fourth order" in the paper's sense; with zero interior knots this reproduces their polynomial.
SPLINE_DEGREE = 4

#: A monotone orbit segment wider than this in phi1 makes the 360 deg branch of a star ambiguous.
_MAX_TRACK_SPAN_DEG = 355.0


def dm_from_distance(d_kpc: np.ndarray) -> np.ndarray:
    """Distance modulus from heliocentric distance in kpc."""
    return 5.0 * np.log10(np.asarray(d_kpc, dtype=float)) + 10.0


def distance_from_dm(dm: np.ndarray) -> np.ndarray:
    """Heliocentric distance in kpc from distance modulus. Positive by construction, which is why
    the correction is fitted in distance modulus and not in distance."""
    return 10.0 ** ((np.asarray(dm, dtype=float) - 10.0) / 5.0)


def _wrap180(x: np.ndarray) -> np.ndarray:
    """Wrap an angular difference in degrees into ``(-180, 180]``."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def _unwrap_deg(x: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(np.asarray(x, dtype=float))))


def _monotone_segment(y: np.ndarray, i0: int) -> Tuple[int, int]:
    """Inclusive knot bounds of the maximal run of constant slope sign in ``y`` containing ``i0``."""
    d = np.diff(y)
    s = np.sign(d)
    s[s == 0] = 1.0  # a flat step joins whichever run precedes it; harmless at 0.2 Myr spacing
    k = int(np.clip(i0, 0, len(s) - 1))
    starts = np.concatenate(([0], np.flatnonzero(s[1:] != s[:-1]) + 1, [len(s)]))
    r = int(np.searchsorted(starts, k, side="right") - 1)
    return int(starts[r]), int(starts[r + 1])


@dataclass
class OrbitTrack:
    """The progenitor orbit in observation space, restricted to one monotone branch of phi1.

    ``phi1`` is strictly increasing and unwrapped (so it can run outside ``(-180, 180]``); ``obs``
    holds :data:`OBS_NAMES` per knot, with ``ra`` likewise unwrapped so it can be interpolated
    across the 0/360 branch cut.
    """

    phi1: np.ndarray
    obs: np.ndarray
    t_gyr_span: float
    frac_knots_used: float

    @property
    def span(self) -> float:
        return float(self.phi1[-1] - self.phi1[0])

    def covers(self, lo: float, hi: float) -> bool:
        return bool(self.phi1[0] <= lo and self.phi1[-1] >= hi)

    def interp(self, phi1: np.ndarray) -> np.ndarray:
        """``(n, 6)`` orbit observables at the requested phi1; NaN outside the branch."""
        phi1 = np.asarray(phi1, dtype=float)
        out = np.full((phi1.size, len(OBS_NAMES)), np.nan)
        inside = (phi1 >= self.phi1[0]) & (phi1 <= self.phi1[-1])
        if inside.any():
            for c in range(len(OBS_NAMES)):
                out[inside, c] = np.interp(phi1[inside], self.phi1, self.obs[:, c])
        return out


def orbit_track(agama, pot, posvel_sat: np.ndarray, frame, R: np.ndarray, T: float, n_knots: int) -> OrbitTrack:
    """Integrate the progenitor over ``[-T, +T]`` and express it in the stream frame.

    ``frame`` is the row's solar frame (``stream_agama._solar_frame``); the projection goes through
    the SAME AGAMA path the simulator uses for the particles, so the orbit and the stars it carries
    cannot desynchronise.
    """
    t, orb = orbit_window(agama, pot, posvel_sat, T, int(n_knots))
    proj = sky_projection(orb[None], np.asarray(frame, dtype=float)[None])[0]
    phi1, _, mu1, mu2 = _sf.project(R, proj[:, 0], proj[:, 1], proj[:, 3], proj[:, 4])

    i0 = int(np.argmin(np.abs(np.asarray(t))))
    # These orbits wrap several times over +-T, so the unwrapped phi1 accumulates into the hundreds
    # of degrees. Anchor it so the present-day knot carries the progenitor's own wrapped phi1 --
    # otherwise the orbit and the stars (branched around the progenitor) live on different absolute
    # scales, and no window covers anything.
    phi1u = _unwrap_deg(phi1)
    phi1u -= 360.0 * np.round((phi1u[i0] - phi1[i0]) / 360.0)

    lo, hi = _monotone_segment(phi1u, i0)
    # Then keep ONE wrap about the present day: a monotone run spanning >360 deg maps several orbit
    # points onto the same sky position, so a star's phi1 would not identify a unique orbit phase.
    keep = np.zeros(phi1u.size, dtype=bool)
    keep[lo : hi + 1] = True
    keep &= np.abs(phi1u - phi1u[i0]) <= _MAX_TRACK_SPAN_DEG / 2.0
    sl = np.flatnonzero(keep)
    sl = slice(int(sl[0]), int(sl[-1]) + 1)

    p = phi1u[sl]
    obs = np.column_stack(
        [
            _unwrap_deg(proj[:, 0])[sl],
            proj[sl, 1],
            dm_from_distance(proj[sl, 2]),
            mu1[sl],
            mu2[sl],
            proj[sl, 5],
        ]
    )
    if p[0] > p[-1]:  # np.interp needs an increasing abscissa
        p, obs = p[::-1], obs[::-1]
    return OrbitTrack(
        phi1=np.ascontiguousarray(p),
        obs=np.ascontiguousarray(obs),
        t_gyr_span=float(np.asarray(t)[sl][-1] - np.asarray(t)[sl][0]),
        frac_knots_used=float((sl.stop - sl.start) / len(phi1u)),
    )


def orbit_track_covering(
    agama, pot, posvel_sat, frame, R, lo: float, hi: float, T0: float,
    knot_dt: float, growth: float = 1.5, max_grow: int = 5,
) -> OrbitTrack:
    """Grow the orbit window until its monotone phi1 branch covers ``[lo, hi]``.

    A fixed window cannot serve the whole prior: a lighter halo lengthens the orbital period, so the
    same +-T covers less phi1. Growing per row costs a few extra millisecond-scale integrations and
    keeps the surrogate honest at the edges of the prior instead of silently truncating the stream
    there. ``knot_dt`` fixes the knot spacing so resolution does not degrade as the window grows.
    """
    T = float(T0)
    track = None
    for _ in range(int(max_grow) + 1):
        n_knots = max(int(round(2.0 * T / knot_dt)) + 1, 101)
        track = orbit_track(agama, pot, posvel_sat, frame, R, T, n_knots)
        if track.covers(lo, hi) or track.span >= _MAX_TRACK_SPAN_DEG:
            return track
        T *= growth
    return track


# --------------------------------------------------------------------------------------------
# spline fitting
# --------------------------------------------------------------------------------------------


def fit_offset_spline(
    x: np.ndarray, y: np.ndarray, x_lo: float, x_hi: float,
    n_interior_knots: int = 3, w: np.ndarray | None = None, guard: float = 5.0,
) -> BSpline:
    """Least-squares degree-4 spline of ``y`` on ``x``, with knots spanning ``[x_lo, x_hi]``.

    ``n_interior_knots = 0`` gives a single quartic polynomial over the whole range -- exactly
    Ibata's "fourth order polynomial fit". Interior knots are placed UNIFORMLY rather than at
    quantiles of ``x``: these streams have a strong progenitor clump, so quantile knots pile up
    inside it, the least-squares system goes near-singular, and the fit develops excursions of many
    orders of magnitude BETWEEN the fitted points while still passing through them. That failure is
    invisible to any median-based check, so it is guarded twice -- uniform knots, and a rejection
    test on a dense grid that backs off a knot and refits until the spline stays within ``guard``
    times the range of the data it is fitting.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    k = SPLINE_DEGREE
    order = np.argsort(x)
    x, y = x[order], y[order]
    if w is not None:
        w = np.asarray(w, dtype=float)[order]

    dense = np.linspace(x_lo, x_hi, 512)
    scale = float(np.max(y) - np.min(y)) if y.size else 0.0
    lim = abs(float(np.median(y))) + guard * max(scale, 1e-12)

    n_int = int(max(0, n_interior_knots))
    while True:
        interior = (
            np.linspace(x_lo, x_hi, n_int + 2)[1:-1] if n_int > 0 else np.empty(0)
        )
        t = np.concatenate([np.full(k + 1, x_lo), interior, np.full(k + 1, x_hi)])
        ok = x.size >= n_int + k + 1 and _knot_intervals_occupied(t, x, k)
        if ok:
            try:
                sp = make_lsq_spline(x, y, t, k=k, w=w)
                v = sp(dense)
                if np.all(np.isfinite(v)) and np.max(np.abs(v)) <= lim:
                    return sp
            except (ValueError, np.linalg.LinAlgError):
                pass
        if n_int == 0:
            # Degree-4 with no interior knots is an ordinary quartic least squares; if even that is
            # unusable the data are degenerate, so fall back to the flat median.
            try:
                t = np.concatenate([np.full(k + 1, x_lo), np.full(k + 1, x_hi)])
                sp = make_lsq_spline(x, y, t, k=k, w=w)
                if np.all(np.isfinite(sp(dense))) and np.max(np.abs(sp(dense))) <= lim:
                    return sp
            except (ValueError, np.linalg.LinAlgError):
                pass
            c = np.full(k + 1, float(np.median(y)))
            return BSpline(np.concatenate([np.full(k + 1, x_lo), np.full(k + 1, x_hi)]), c, k)
        n_int -= 1


def _knot_intervals_occupied(t: np.ndarray, x: np.ndarray, k: int) -> bool:
    """Schoenberg-Whitney in the form that matters here: no empty interior knot span."""
    edges = np.unique(t[k : len(t) - k])
    if edges.size < 2:
        return True
    counts, _ = np.histogram(x, bins=edges)
    return bool(np.all(counts > 0))


def _binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray, fn, min_count: int):
    """Per-bin statistic with the bin centre of mass as abscissa; bins below ``min_count`` dropped."""
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    cx, cy, cn = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n < min_count:
            continue
        v = fn(y[m])
        if not np.isfinite(v):
            continue
        cx.append(float(np.mean(x[m])))
        cy.append(float(v))
        cn.append(n)
    return np.asarray(cx), np.asarray(cy), np.asarray(cn, dtype=float)


def eval_spline(sp: BSpline, x: np.ndarray) -> np.ndarray:
    """Evaluate a fitted spline, clipping ``x`` into its knot span.

    Degree-4 extrapolation is violent, and the knots only span the phi1 range where the fiducial
    stream actually had stars in every bin. Clipping holds the correction flat beyond that instead,
    which is the conservative choice at the sparse ends of a stream.
    """
    k = sp.k
    return sp(np.clip(np.asarray(x, dtype=float), sp.t[k], sp.t[-k - 1]))


def _mad_std(v: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


# --------------------------------------------------------------------------------------------
# the template
# --------------------------------------------------------------------------------------------


@dataclass
class OffsetTemplate:
    """Everything needed to paint one stream onto an arbitrary progenitor orbit."""

    R: np.ndarray
    T0: float
    knot_dt: float
    delta: Dict[str, BSpline]
    sigma: Dict[str, BSpline]
    sigma_bounds: Dict[str, Tuple[float, float]]
    phi1_cdf_q: np.ndarray
    phi1_cdf_x: np.ndarray
    phi1_lo: float
    phi1_hi: float
    diagnostics: Dict[str, float] = field(default_factory=dict)


def build_offset_template(
    agama, pot_fid, posvel_sat, frame_fid, xv_stars: np.ndarray, R: np.ndarray,
    T0: float, knot_dt: float, n_interior_knots: int = 3, n_bins: int = 40,
    min_count: int = 5, n_quantiles: int = 1001,
) -> OffsetTemplate:
    """Fit DeltaTheta(phi1), sigma_Theta(phi1) and the phi1 density from one simulated stream.

    ``xv_stars`` is the fiducial run's Galactocentric cloud. It is projected here rather than read
    from the stored ``sim_data_projected`` so that both sides of the residual use the same AGAMA
    projection path the simulator will use later.
    """
    proj = sky_projection(np.asarray(xv_stars, dtype=float)[None], np.asarray(frame_fid, float)[None])[0]
    proj = proj[np.isfinite(proj).all(axis=1)]
    if proj.shape[0] < n_bins * min_count:
        raise ValueError(f"only {proj.shape[0]} finite stars for {n_bins} bins")

    phi1_raw, _, mu1_s, mu2_s = _sf.project(R, proj[:, 0], proj[:, 1], proj[:, 3], proj[:, 4])

    # Branch the stars around the progenitor's own phi1: a stream spans well under 360 deg, so
    # "within +-180 deg of the progenitor" is unambiguous and needs no orbit yet.
    prog_proj = sky_projection(np.asarray(posvel_sat, dtype=float)[None, None], np.asarray(frame_fid, float)[None])[0, 0]
    phi1_prog = float(_sf.project(R, prog_proj[None, 0], prog_proj[None, 1], prog_proj[None, 3], prog_proj[None, 4])[0][0])
    phi1_s = _sf.unwrap_to(phi1_raw, np.full(phi1_raw.shape, phi1_prog))

    lo, hi = float(phi1_s.min()), float(phi1_s.max())
    if hi - lo >= _MAX_TRACK_SPAN_DEG:
        raise ValueError(f"stream spans {hi - lo:.1f} deg in phi1; the 360 deg branch is ambiguous")

    track = orbit_track_covering(agama, pot_fid, posvel_sat, frame_fid, R, lo, hi, T0, knot_dt)
    if not track.covers(lo, hi):
        raise ValueError(
            f"orbit phi1 branch [{track.phi1[0]:.1f}, {track.phi1[-1]:.1f}] does not cover the "
            f"stream [{lo:.1f}, {hi:.1f}]; raise --T-gyr"
        )

    star = np.column_stack(
        [_unwrap_deg(proj[:, 0]), proj[:, 1], dm_from_distance(proj[:, 2]), mu1_s, mu2_s, proj[:, 5]]
    )
    base = track.interp(phi1_s)
    resid = star - base
    resid[:, 0] = _wrap180(resid[:, 0])  # ra is periodic; the others are not

    # Uniform in phi1, NOT equal-count: the fit needs its data spread over the whole range, and
    # equal-count bins would put nearly all of them inside the progenitor clump.
    edges = np.linspace(lo, hi, n_bins + 1)
    delta: Dict[str, BSpline] = {}
    sigma: Dict[str, BSpline] = {}
    sigma_bounds: Dict[str, Tuple[float, float]] = {}
    diag: Dict[str, float] = {}
    for c, name in enumerate(OBS_NAMES):
        bx, by, bn = _binned(phi1_s, resid[:, c], edges, np.median, min_count)
        if bx.size < SPLINE_DEGREE + 1:
            raise ValueError(f"{name}: only {bx.size} usable phi1 bins")
        # Knots span the OCCUPIED range so nothing is ever extrapolated; sample_stream clips to it.
        klo, khi = float(bx[0]), float(bx[-1])
        delta[name] = fit_offset_spline(bx, by, klo, khi, n_interior_knots, w=bn)
        scatter = resid[:, c] - delta[name](np.clip(phi1_s, klo, khi))
        sx, sy, sn = _binned(phi1_s, scatter, edges, _mad_std, max(min_count, 5))
        floor = max(1e-3 * float(np.median(sy)), 1e-12)
        sigma[name] = fit_offset_spline(
            sx, np.log(np.maximum(sy, floor)), float(sx[0]), float(sx[-1]), n_interior_knots,
            w=sn, guard=2.0,
        )
        # The width is never allowed outside the range actually measured in the fiducial. A
        # log-sigma spline that overshoots by a few in log space is a factor of hundreds in sigma,
        # and the excursion guard cannot be tight enough in log space to catch that on its own.
        sigma_bounds[name] = (float(np.min(sy)), float(np.max(sy)))
        diag[f"resid_mad_{name}"] = _mad_std(resid[:, c])
        diag[f"after_fit_mad_{name}"] = _mad_std(scatter)
        diag[f"delta_absmax_{name}"] = float(np.max(np.abs(by)))

    diag["n_stars_fitted"] = float(proj.shape[0])
    diag["phi1_span_deg"] = hi - lo
    diag["track_span_deg"] = track.span
    diag["track_frac_knots_used"] = track.frac_knots_used
    diag["n_interior_knots"] = float(n_interior_knots)

    q = np.linspace(0.0, 1.0, int(n_quantiles))
    tpl = OffsetTemplate(
        R=np.asarray(R, dtype=float),
        T0=float(T0),
        knot_dt=float(knot_dt),
        delta=delta,
        sigma=sigma,
        sigma_bounds=sigma_bounds,
        phi1_cdf_q=q,
        phi1_cdf_x=np.quantile(phi1_s, q),
        phi1_lo=lo,
        phi1_hi=hi,
        diagnostics=diag,
    )
    return tpl


# --------------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------------


def sample_stream(tpl: OffsetTemplate, track: OrbitTrack, n_stars: int, rng: np.random.Generator):
    """``(n, 6)`` ICRS observables (ra, dec, distance, mu_ra_cosdec, mu_dec, v_los).

    Returns ``(obs, phi1, miss_frac)``. ``miss_frac`` is the fraction of the template's stream that
    this potential's orbit does not reach in phi1; the stars actually drawn are the remainder, so
    the emitted cloud is the stream CONDITIONED on being within reach rather than a cloud with holes
    in it. That matters because ``stream_agama.window_subsample`` discards an entire row if a single
    star is non-finite -- scattering NaNs through the sample would silently delete rows near the
    edge of the prior, which is exactly where a surrogate's behaviour needs to stay visible.
    """
    a = max(float(tpl.phi1_cdf_x[0]), float(track.phi1[0]))
    b = min(float(tpl.phi1_cdf_x[-1]), float(track.phi1[-1]))
    n = int(n_stars)
    if not b > a:
        return np.full((n, 6), np.nan), np.full(n, np.nan), 1.0

    ua = float(np.interp(a, tpl.phi1_cdf_x, tpl.phi1_cdf_q))
    ub = float(np.interp(b, tpl.phi1_cdf_x, tpl.phi1_cdf_q))
    miss = float(1.0 - (ub - ua))
    phi1 = np.interp(ua + (ub - ua) * rng.random(n), tpl.phi1_cdf_q, tpl.phi1_cdf_x)

    base = track.interp(phi1)
    vals = np.empty_like(base)
    for c, name in enumerate(OBS_NAMES):
        s_lo, s_hi = tpl.sigma_bounds[name]
        sig = np.clip(np.exp(eval_spline(tpl.sigma[name], phi1)), s_lo, s_hi)
        vals[:, c] = base[:, c] + eval_spline(tpl.delta[name], phi1) + rng.normal(size=phi1.size) * sig

    ra = vals[:, 0] % 360.0
    dec = np.clip(vals[:, 1], -89.999, 89.999)
    dist = distance_from_dm(vals[:, 2])
    # The proper-motion correction lives in the stream frame, so it has to be de-projected at the
    # star's OWN (ra, dec) -- which the position correction has just moved off the phi1 it was
    # evaluated at. This is Ibata's own parameterization; the induced |dphi1| is a build diagnostic.
    p1, p2, _, _ = _sf.project(tpl.R, ra, dec, np.zeros_like(ra), np.zeros_like(ra))
    _, _, mura, mudec = _sf.deproject(tpl.R, p1, p2, vals[:, 3], vals[:, 4])

    obs = np.column_stack([ra, dec, dist, mura, mudec, vals[:, 5]])
    return obs, phi1, miss


def observables_to_cartesian(agama, obs: np.ndarray, frame) -> np.ndarray:
    """ICRS observables ``(n, 6)`` -> Galactocentric ``(n, 6)``, the inverse of ``sky_projection``.

    The same two AGAMA calls ``stream_agama._simulate_one`` uses for the progenitor, vectorized.
    ``simulate`` re-projects the result, and that round trip is exact to ~1e-13 (pinned by a test).
    """
    obs = np.asarray(obs, dtype=float)
    out = np.full_like(obs, np.nan)
    ok = np.isfinite(obs).all(axis=1)
    if not ok.any():
        return out
    o = obs[ok]
    lon, lat, pml, pmb = agama.transformCelestialCoords(
        agama.fromICRStoGalactic, np.radians(o[:, 0]), np.radians(o[:, 1]), o[:, 3], o[:, 4]
    )
    xv = np.asarray(
        agama.getGalactocentricFromGalactic(
            lon, lat, o[:, 2], pml * 4.74, pmb * 4.74, o[:, 5],
            galcen_distance=float(frame[0]), galcen_v_sun=tuple(float(v) for v in frame[1:4]),
            z_sun=float(frame[4]),
        )
    ).T
    out[ok] = xv
    return out
