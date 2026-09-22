#!/usr/bin/env python
"""The published STREAMFINDER (Ibata et al. 2024, ApJ, Table 3) stream coordinate frames.

Columns (4)-(6) of that table are, verbatim from its note, "the zero-point in R.A. and position of
the pole of the coordinate system used to derive the phi_1 and phi_2 stream coordinates" — so a
stream's frame is fully specified by ``(alpha_0, alpha_pole, delta_pole)`` and reproducing it needs
no fit. This is what lets our panels be read against the published ones; the alternative in this
repo, ``ppc_summary_statistics.fit_frame``, fits a great circle to whichever member set is loaded,
which gives a frame whose pole, handedness and zero-point all move with the catalogue.

Frame construction, given the pole ``p`` and the zero-point RA ``alpha_0``:

    x0  the point of the great circle (n . p = 0) lying on the meridian RA = alpha_0, i.e.
        x0 ~ p x m with m = (-sin a0, cos a0, 0) the normal of that meridian plane, signed so that
        RA(x0) = alpha_0 rather than alpha_0 + 180 deg
    y0 = p x x0
    R  = [x0, y0, p]      ->   phi1 = atan2(n.y0, n.x0),  phi2 = asin(n.p)

so phi1 = 0 on the stream's own zero-point meridian and phi2 = 0 on its great circle, both as
published. Note ``alpha_0`` is the progenitor's RA for the three globular-cluster streams here
(Pal-5 229.022, Gjoll 154.403, Fjorm 189.867), but phi1 = 0 is the point of the GREAT CIRCLE at that
RA, which is not the progenitor itself unless the progenitor sits exactly on the circle.

This project's stream names differ from the table's: Pal5 -> "Pal-5", NGC3201 -> "Gjoll",
M68 -> "Fjorm" (:data:`TABLE_NAME`).
"""

from __future__ import annotations

import os

import numpy as np

DEFAULT_TABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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


if __name__ == "__main__":
    tab = read_table()
    for name, tname in TABLE_NAME.items():
        r = tab[tname]
        print(f"{name:8s} = {tname:8s} alpha_0 {r['alpha_0']:8.3f}  pole "
              f"({r['alpha_pole']:7.3f}, {r['delta_pole']:7.3f})  n {r['n']:4d}  n_v {r['n_v']:3d}")


# --------------------------------------------------------------------------------------------
# Palau & Miralda-Escude's frame construction (their Appendix A2)
# --------------------------------------------------------------------------------------------
# Their R1 is defined by two steps, both of which we can reproduce from the member selection:
#
#   1. the pole minimises the SCATTER of phi2 over the stream stars,
#         L1(ax, ay) = sum (phi2 - mean(phi2))^2,
#      which for small phi2 is the smallest-eigenvalue eigenvector of the COVARIANCE of the member
#      unit vectors (centred). Note this leaves mean(phi2) free — their R1 indeed has a non-zero
#      mean phi2, and they give R2 separately for the usual L2 = sum phi2^2 convention, whose pole
#      is the smallest eigenvector of the UNCENTRED scatter matrix (== `fit_frame`'s pole).
#   2. "a horizontal rotation of az ... to position the closest extreme of the stream to the cluster
#      at the origin of the coordinates", i.e. phi1 = 0 at the end of the member distribution
#      nearest the progenitor, with phi1 increasing away from it.
#
# Reproducing this on our member files gives, for M68/Fjorm, progenitor phi1 = -24.1 deg (L1, full
# 287-star Palau selection) / -23.9 (main component), matching the value quoted from their figures;
# the recovered pole (RA 92.1, Dec 11.8) agrees with the pole implied by their published R1 to a few
# degrees, the residual being their full 291-star selection vs our in-window subset.
#
# Their printed R1/R2 matrices are NOT used: transcribed from the PDF they are orthonormal but their
# pole lies ~34 deg from the M68 members' own great circle, which cannot be right, so the numbers are
# taken to be garbled. The construction above is unambiguous and is used instead.


def palau_frame(ra, dec, prog_ra, prog_dec, loss: str = "L1") -> np.ndarray:
    """Palau & Miralda-Escude Appendix A2 frame from a member selection and its progenitor."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    n = np.stack(_unit(ra, dec), axis=-1) if ra.ndim == 0 else np.stack(
        [np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
         np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
         np.sin(np.radians(dec))], axis=-1)
    ncl = _unit(prog_ra, prog_dec)
    scatter = np.cov(n.T) if loss == "L1" else n.T @ n
    pole = np.linalg.eigh(scatter)[1][:, 0]
    x = ncl - (ncl @ pole) * pole
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(pole, x), pole], axis=0)
    phi1 = np.degrees(np.arctan2(n @ R[1], n @ R[0]))
    if abs(phi1.max()) < abs(phi1.min()):          # orient phi1 to increase away from the progenitor
        R = np.stack([x, -np.cross(pole, x), pole], axis=0)
        phi1 = -phi1
    end = np.radians(phi1.min())                   # stream extreme closest to the progenitor -> 0
    c, s = np.cos(end), np.sin(end)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ R


# The canonical member selection Palau & Miralda-Escude fit their frame to. The frame is a fixed,
# published thing, so it is ALWAYS built from this file — it must not move when a different overlay
# catalogue is plotted.
PALAU_SELECTION = os.path.join(
    os.path.dirname(DEFAULT_TABLE),
    "gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau.npz",
)
PROGENITOR_RADEC = {"Pal5": (229.022, -0.112), "NGC3201": (154.403, -46.412),
                    "M68": (189.867, -26.744)}


def _members(path: str, j: int):
    d = np.load(path)
    rs = d["sim_data_projected"]
    rs = rs[0] if rs.ndim == 4 else rs
    am = d["attention_mask"]
    am = am[:, 0, :] if am.ndim == 3 else am
    jj = np.asarray(d["j"]).reshape(-1).astype(int)
    i = int(np.where(jj == j)[0][0])
    m = rs[i][am[i].astype(bool)]
    return m[:, 0], m[:, 1]


def palau_m68_frame(loss: str = "L1", path: str = PALAU_SELECTION) -> np.ndarray:
    """M68/Fjorm in Palau & Miralda-Escude's frame (progenitor at phi1 = -24.1 deg with L1)."""
    ra, dec = _members(path, 2)
    return palau_frame(ra, dec, *PROGENITOR_RADEC["M68"], loss=loss)


def mixed_frames(loss: str = "L1") -> dict[str, np.ndarray]:
    """Palau's frame for M68 (the stream their appendix defines it for), the published STREAMFINDER
    frames for Pal5 and NGC3201.

    Palau's L1 pole minimises only the SCATTER of phi2, which is degenerate for a short stream: run
    on Pal5's 30-deg-long member list it returns a pole nearly in the stream's own plane, giving a
    ~59 deg constant phi2 offset. So it is applied only to M68, for which it was derived.
    """
    out = frames()
    out["M68"] = palau_m68_frame(loss=loss)
    return out
