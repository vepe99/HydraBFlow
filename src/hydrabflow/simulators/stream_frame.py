"""Great-circle stream frames: (ra, dec, pm) <-> (phi1, phi2, mu_phi1, mu_phi2).

A stream frame is a rotation ``R`` whose rows are ``(x0, y0, pole)``: phi1 runs along the great
circle from the zero-point ``x0``, phi2 is the angle off it, and the published STREAMFINDER frames
(Ibata et al. 2024, Table 3) are built from the tabulated pole and RA zero-point by
``frame`` / ``frames`` below (``scripts/streamfinder_frame.py`` re-exports them).

``project`` is a verbatim lift of ``scripts/ppc_summary_statistics.project`` -- the math the whole
PPC stack and the training summary statistics already use -- so that a simulator working in the
stream frame and the diagnostics judging it share one definition. The duplication is deliberate:
``ppc_summary_statistics`` is load-bearing for a dozen scripts and is left untouched;
``tests/test_stream_orbit_offset.py`` pins the two against each other to machine precision.

Conventions, all of which matter and none of which are visible from the signatures:

* ``mu_ra_cosdec`` already carries the ``cos(dec)`` factor, and ``mu_phi1`` likewise carries
  ``cos(phi2)`` -- both are the "true angular rate on the sky", so the pair transforms as an
  ordinary tangent vector and no extra cosine appears anywhere below.
* The proper motion is turned into a 3-D tangent vector in ICRS, rotated by the SAME ``R`` as the
  position, and re-projected onto the frame's tangent basis. There is no small-angle step, so the
  transform is exact anywhere on the sphere.
* phi1 comes out of ``arctan2`` and so lands in ``(-180, 180]``. NGC 3201 and M68 span more than
  100 deg, so any along-track interpolation has to work on an unwrapped phi1 -- see ``unwrap_to``.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["unit_vec", "project", "deproject", "unwrap_to", "frame", "frames", "read_table"]

DEFAULT_TABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "assets", "gaia", "stream_detected_streamfinder.ascii",
)
# this project's name -> the name in Ibata+2024 Table 3
TABLE_NAME = {"Pal5": "Pal-5", "NGC3201": "Gjoll", "M68": "Fjorm"}


def _unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def read_table(path: str = DEFAULT_TABLE) -> dict[str, dict]:
    """{table name: {alpha_0, alpha_pole, delta_pole, n, n_v}} for every row of Table 3."""
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.rstrip("\n").split("\t")]
            if len(parts) < 8 or not parts[0].isdigit():
                continue
            try:
                a0, ap, dp = (float(parts[3]), float(parts[4]), float(parts[5]))
            except ValueError:
                continue
            out[parts[1]] = dict(alpha_0=a0, alpha_pole=ap, delta_pole=dp,
                                 n=int(parts[6]), n_v=int(parts[7]))
    return out


def frame(alpha_0: float, alpha_pole: float, delta_pole: float) -> np.ndarray:
    """Rotation matrix ``R`` (rows x0, y0, pole) for one published frame."""
    p = _unit(alpha_pole, delta_pole)
    a0 = np.radians(alpha_0)
    m = np.array([-np.sin(a0), np.cos(a0), 0.0])   # normal of the RA = alpha_0 meridian plane
    x0 = np.cross(p, m)
    n = np.linalg.norm(x0)
    if n < 1e-12:                                   # pole on the meridian: any circle point will do
        raise ValueError("pole lies in the alpha_0 meridian plane; frame is degenerate")
    x0 /= n
    # the two intersections differ by 180 deg in RA; keep the one at alpha_0
    if np.cos(np.arctan2(x0[1], x0[0]) - a0) < 0:
        x0 = -x0
    return np.stack([x0, np.cross(p, x0), p], axis=0)


def frames(names=("Pal5", "NGC3201", "M68"), path: str = DEFAULT_TABLE) -> dict[str, np.ndarray]:
    """{this project's stream name: R} for the requested streams."""
    tab = read_table(path)
    out = {}
    for name in names:
        row = tab[TABLE_NAME[name]]
        out[name] = frame(row["alpha_0"], row["alpha_pole"], row["delta_pole"])
    return out



def unit_vec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Unit vectors ``(..., 3)`` from spherical angles in degrees."""
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def _tangent_basis(a: np.ndarray, d: np.ndarray):
    """The orthonormal tangent pair ``(e_a, e_d)`` at angles given in RADIANS."""
    e = np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], -1)
    m = np.stack([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)], -1)
    return e, m


def project(R: np.ndarray, ra, dec, mura, mudec):
    """ICRS -> stream frame. Returns ``(phi1, phi2, mu_phi1, mu_phi2)``, degrees and mas/yr."""
    ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    mura, mudec = np.asarray(mura, dtype=float), np.asarray(mudec, dtype=float)
    n = unit_vec(ra, dec)
    e, m = _tangent_basis(np.radians(ra), np.radians(dec))
    v = mura[:, None] * e + mudec[:, None] * m
    npr, vpr = n @ R.T, v @ R.T
    phi1 = np.degrees(np.arctan2(npr[:, 1], npr[:, 0]))
    phi2 = np.degrees(np.arcsin(np.clip(npr[:, 2], -1, 1)))
    ep, mp = _tangent_basis(np.radians(phi1), np.radians(phi2))
    return phi1, phi2, np.sum(vpr * ep, 1), np.sum(vpr * mp, 1)


def deproject(R: np.ndarray, phi1, phi2, mu_phi1, mu_phi2):
    """Stream frame -> ICRS, the exact inverse of :func:`project`.

    ``ra`` is wrapped to ``[0, 360)`` to match ``stream_common._sky_projection_agama``.
    """
    phi1, phi2 = np.asarray(phi1, dtype=float), np.asarray(phi2, dtype=float)
    mu_phi1 = np.asarray(mu_phi1, dtype=float)
    mu_phi2 = np.asarray(mu_phi2, dtype=float)

    npr = unit_vec(phi1, phi2)
    ep, mp = _tangent_basis(np.radians(phi1), np.radians(phi2))
    vpr = mu_phi1[:, None] * ep + mu_phi2[:, None] * mp

    # rows of R are the frame axes in ICRS, so ``npr = n @ R.T`` inverts as ``n = npr @ R``.
    n, v = npr @ R, vpr @ R
    ra = np.degrees(np.arctan2(n[:, 1], n[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(n[:, 2], -1, 1)))
    e, m = _tangent_basis(np.radians(ra), np.radians(dec))
    return ra, dec, np.sum(v * e, 1), np.sum(v * m, 1)


def unwrap_to(phi1: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Add the multiple of 360 deg that brings each ``phi1`` closest to its ``reference``.

    ``project`` returns phi1 in ``(-180, 180]``; a stream longer than that, or one straddling the
    branch cut, then looks discontinuous. Given a per-element reference on the continuous
    (``np.unwrap``-ed) orbit track, this puts every star on the same branch as the orbit.
    """
    phi1 = np.asarray(phi1, dtype=float)
    return phi1 + 360.0 * np.round((np.asarray(reference, dtype=float) - phi1) / 360.0)
