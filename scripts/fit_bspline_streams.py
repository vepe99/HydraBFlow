"""Weighted cubic B-spline fits of the observed stream members vs phi1.

Standalone (numpy + scipy + matplotlib). Frame = the published Ibata+2024 STREAMFINDER
poles (``streamfinder_frame.frames``); observables = phi2, parallax, mu_phi1, mu_phi2 and
v_los (MEASURED stars only), each fitted separately as ``BSpline(phi1)`` with weights
1/sigma from ``obs_error`` and a bootstrap 68 % band.

    .venv/bin/python scripts/fit_bspline_streams.py \
        --npz assets/gaia/gaia_observed_streams_palau23_dr3.npz --out outputs/Bsline/palau23_dr3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamfinder_frame import frames  # noqa: E402

STREAMS = ["Pal5", "NGC3201", "M68"]
OBS = [("phi2", "phi2 [deg]"), ("parallax", "parallax [mas]"), ("mu_phi1", "mu_phi1 [mas/yr]"),
       ("mu_phi2", "mu_phi2 [mas/yr]"), ("vlos", "v_los [km/s]")]
K = 3  # cubic


def project(R, ra, dec, mura, mudec):
    """ICRS -> stream frame (copy of simulators.stream_frame.project; keeps this script import-free)."""
    a, d = np.radians(ra), np.radians(dec)
    n = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], -1)
    e = np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], -1)
    m = np.stack([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)], -1)
    v = mura[:, None] * e + mudec[:, None] * m
    npr, vpr = n @ R.T, v @ R.T
    phi1 = np.degrees(np.arctan2(npr[:, 1], npr[:, 0]))
    phi2 = np.degrees(np.arcsin(np.clip(npr[:, 2], -1, 1)))
    p1, p2 = np.radians(phi1), np.radians(phi2)
    ep = np.stack([-np.sin(p1), np.cos(p1), np.zeros_like(p1)], -1)
    mp = np.stack([-np.sin(p2) * np.cos(p1), -np.sin(p2) * np.sin(p1), np.cos(p2)], -1)
    return phi1, phi2, np.sum(vpr * ep, 1), np.sum(vpr * mp, 1)


def knots(x, n_interior):
    """Clamped cubic knot vector with interior knots at equal-count quantiles of x."""
    lo, hi = x.min(), x.max()
    inner = np.quantile(x, np.linspace(0, 1, n_interior + 2)[1:-1]) if n_interior else np.array([])
    return np.r_[[lo] * (K + 1), inner, [hi] * (K + 1)]


def fit(x, y, sig, n_interior, n_boot=300, rng=None):
    o = np.argsort(x)
    x, y, w = x[o], y[o], 1.0 / sig[o]
    t = knots(x, n_interior)
    spl = make_lsq_spline(x, y, t, k=K, w=w)
    res = y - spl(x)
    dof = max(len(x) - (len(t) - K - 1), 1)
    boots = []
    rng = rng or np.random.default_rng(0)
    for _ in range(n_boot):
        i = np.sort(rng.integers(0, len(x), len(x)))
        xi = x[i]
        if xi[0] > t[0] or xi[-1] < t[-1]:  # keep the clamped ends inside the data
            xi = np.r_[x[0], xi, x[-1]]
            yi, wi = np.r_[y[0], y[i], y[-1]], np.r_[w[0], w[i], w[-1]]
        else:
            yi, wi = y[i], w[i]
        try:
            boots.append(make_lsq_spline(xi, yi, t, k=K, w=wi).c)
        except (ValueError, np.linalg.LinAlgError):
            pass
    return spl, np.array(boots), dict(n=int(len(x)), n_interior_knots=int(n_interior),
                                       rms=float(np.sqrt(np.mean(res**2))),
                                       chi2_dof=float(np.sum((res * w) ** 2) / dof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="assets/gaia/gaia_observed_streams_palau23_dr3.npz")
    ap.add_argument("--out", default="outputs/Bsline/palau23_dr3")
    ap.add_argument("--stars-per-knot", type=int, default=15, help="interior knots = N // this, in [1, 8]")
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.npz)
    X, M = d["sim_data_projected"][0], d["attention_mask"][:, 0] > 0
    E, V = d["obs_error"][0], d["vlos_mask"] > 0
    R = frames(STREAMS)
    summary, store = {}, {}
    for j, name in enumerate(STREAMS):
        s, e, vm = X[j][M[j]], E[j][M[j]], V[j][M[j]]
        phi1, phi2, mu1, mu2 = project(R[name], s[:, 0], s[:, 1], s[:, 3], s[:, 4])
        # pm errors: propagate variances through the local ICRS->stream rotation angle
        _, _, c, sn = project(R[name], s[:, 0], s[:, 1], np.ones(len(s)), np.zeros(len(s)))
        e1 = np.sqrt((c * e[:, 3]) ** 2 + (sn * e[:, 4]) ** 2)
        e2 = np.sqrt((sn * e[:, 3]) ** 2 + (c * e[:, 4]) ** 2)
        # sky-position errors are ~0 in the table: use the fit's own scatter, i.e. unit weights
        cols = {"phi2": (phi2, np.ones_like(phi2), slice(None)),
                "parallax": (s[:, 2], e[:, 2], slice(None)),
                "mu_phi1": (mu1, e1, slice(None)),
                "mu_phi2": (mu2, e2, slice(None)),
                "vlos": (s[:, 5], e[:, 5], vm)}
        fig, axes = plt.subplots(len(OBS), 1, figsize=(9, 2.6 * len(OBS)), sharex=True)
        summary[name] = {}
        for ax, (key, label) in zip(axes, OBS):
            y, sig, sel = cols[key]
            x, y, sig = phi1[sel], y[sel], np.where(sig[sel] > 0, sig[sel], np.nanmedian(sig[sel]))
            n_int = int(np.clip(len(x) // a.stars_per_knot, 1, 8))
            if len(x) < 2 * (K + 1):
                ax.text(0.5, 0.5, f"{key}: only {len(x)} stars", transform=ax.transAxes, ha="center")
                ax.set_ylabel(label)
                continue
            spl, boots, info = fit(x, y, sig, n_int, a.n_boot, np.random.default_rng(j))
            summary[name][key] = info
            store[f"{name}/{key}/t"], store[f"{name}/{key}/c"] = spl.t, spl.c
            store[f"{name}/{key}/c_boot"] = boots
            g = np.linspace(x.min(), x.max(), 400)
            ax.errorbar(x, y, yerr=None if key == "phi2" else sig, fmt="o", ms=3, lw=0.6,
                        color="0.4", alpha=0.7, label="members")
            if len(boots):
                band = np.array([BSpline(spl.t, c, K)(g) for c in boots])
                ax.fill_between(g, *np.percentile(band, [16, 84], axis=0), color="C3", alpha=0.3,
                                label="bootstrap 68 %")
            ax.plot(g, spl(g), color="C3", lw=2, label=f"B-spline (k=3, {n_int} interior knots)")
            for tk in spl.t[K + 1:-K - 1]:
                ax.axvline(tk, color="C3", lw=0.5, ls=":")
            ax.set_ylabel(label)
            ax.text(0.01, 0.95, f"rms {info['rms']:.3g}  chi2/dof {info['chi2_dof']:.2f}  N={info['n']}",
                    transform=ax.transAxes, va="top", fontsize=8)
        axes[0].legend(fontsize=8, loc="lower right")
        axes[0].set_title(f"{name}: weighted cubic B-spline fits vs phi1 (STREAMFINDER frame)")
        axes[-1].set_xlabel("phi1 [deg]")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{name}_bspline.png"), dpi=140)
        plt.close(fig)

    np.savez(os.path.join(a.out, "bspline_fits.npz"), **store)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
