#!/usr/bin/env python
"""Fit Ibata et al. (2024) DeltaTheta(phi1) corrections from a finished stream simulation.

Input: a grouped npz from ``simulate_multistream`` that stored ``sim_data_carthesian`` (the full
Galactocentric cloud) plus its ``.hydra`` snapshot. Output: one template per stream -- six degree-4
correction splines, six degree-4 log-dispersion splines and the phi1 density -- which
``stream_orbit_offset`` then paints onto the progenitor orbit of any new potential.

The template MUST be built in the potential the stars actually moved in, reconstructed from the
run's own resolved config, otherwise every residual is measured against the wrong orbit.

    .venv/bin/python scripts/build_orbit_offset_template.py \
        --sim data_local/mcmillan17_mean_tend_best/mcmillan17_mean.npz \
        --out data_local/mcmillan17_mean_tend_best/orbit_offset_template.npz

``--n-interior-knots 0`` reproduces Ibata's plain fourth-order polynomial exactly; the default adds
interior knots so the correction can bend where one quartic cannot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from build_trihedron_template import (  # noqa: E402
    NAMES,
    pot_cfg_from_snapshot,
    progenitor_state,
    row_of,
)
from hydrabflow.simulators import orbit_offset as oo  # noqa: E402
from hydrabflow.simulators import stream_frame as sf  # noqa: E402
from hydrabflow.simulators.frozen_locals import FROZEN_LOCAL_KEYS  # noqa: E402
from hydrabflow.simulators.stream_agama import _agama, _host_potential  # noqa: E402
from streamfinder_frame import frames as streamfinder_frames  # noqa: E402
from streamfinder_frame import mixed_frames  # noqa: E402


def stream_frames(kind: str) -> dict:
    """Published (Ibata+2024 Table 3) stream frames, keyed by stream index."""
    table = {"streamfinder": streamfinder_frames, "palau": mixed_frames}[kind]()
    inv = {v: k for k, v in NAMES.items()}
    return {inv[name]: R for name, R in table.items()}


def self_test(agama, pot_fid, posvel, frame, R, tpl, xv_stars, seed=0, n_bins=10):
    """Rebuild the stream in its OWN potential and compare to the simulation it came from.

    Unlike the trihedron there is no exactness anchor -- this model is lossy by construction -- so
    the test is a tolerance on the binned median track and on the per-bin dispersion, and the
    achieved numbers are what the config header quotes.
    """
    from hydrabflow.simulators.stream_common import sky_projection

    track = oo.orbit_track_covering(
        agama, pot_fid, posvel, frame, R, tpl.phi1_lo, tpl.phi1_hi, tpl.T0, tpl.knot_dt
    )
    n = int(np.asarray(xv_stars).shape[0])
    obs, phi1_draw, miss = oo.sample_stream(tpl, track, n, np.random.default_rng(seed))

    truth = sky_projection(np.asarray(xv_stars, float)[None], np.asarray(frame, float)[None])[0]
    truth = truth[np.isfinite(truth).all(1)]
    ok = np.isfinite(obs).all(1)

    prog = sky_projection(np.asarray(posvel, float)[None, None], np.asarray(frame, float)[None])[0, 0]
    p0 = float(sf.project(R, prog[None, 0], prog[None, 1], prog[None, 3], prog[None, 4])[0][0])

    def frame_coords(a):
        p1, p2, m1, m2 = sf.project(R, a[:, 0], a[:, 1], a[:, 3], a[:, 4])
        return sf.unwrap_to(p1, np.full(p1.shape, p0)), p2, m1, m2

    pt, p2t, m1t, m2t = frame_coords(truth)
    ps, p2s, m1s, m2s = frame_coords(obs[ok])

    edges = np.quantile(pt, np.linspace(0, 1, n_bins + 1))
    out = {"phi1_miss_frac": miss}
    for label, (yt, ys) in {
        "phi2": (p2t, p2s),
        "parallax": (1.0 / truth[:, 2], 1.0 / obs[ok, 2]),
        "mu_phi1": (m1t, m1s),
        "mu_phi2": (m2t, m2s),
        "vlos": (truth[:, 5], obs[ok, 5]),
    }.items():
        bt = _binstat(pt, yt, edges, np.median)
        bs = _binstat(ps, ys, edges, np.median)
        st = _binstat(pt, yt, edges, oo._mad_std)
        ss = _binstat(ps, ys, edges, oo._mad_std)
        out[f"track_err_{label}"] = float(np.nanmedian(np.abs(bs - bt)))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"width_ratio_{label}"] = float(np.nanmedian(ss / st))
            # Medians and MADs are blind to a spline that blows up between the points it was fitted
            # to (a real bug during development: log-sigma excursions of 1e290 that every
            # median-based check passed). The 1-99 percentile span is not.
            rt = float(np.diff(np.percentile(yt, [1, 99]))[0])
            rs = float(np.diff(np.percentile(ys, [1, 99]))[0])
            out[f"p99_range_ratio_{label}"] = float(rs / rt) if rt > 0 else float("nan")

    # How far the (ra, dec) correction moves a star off the phi1 it was evaluated at -- the size of
    # the self-inconsistency inherent in Ibata's parameterization.
    out["induced_dphi1_mad_deg"] = float(oo._mad_std(ps - phi1_draw[ok]))
    return out


def _binstat(x, y, edges, fn, min_count=5):
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    return np.array(
        [fn(y[idx == b]) if int((idx == b).sum()) >= min_count else np.nan for b in range(len(edges) - 1)]
    )


def make_figure(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["streams"])
    fig, axes = plt.subplots(len(names), 6, figsize=(21, 3.1 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        d = report["streams"][name]["diagnostics"]
        for c, obs in enumerate(oo.OBS_NAMES):
            ax = axes[r][c]
            ax.bar(
                [0, 1],
                [d[f"resid_mad_{obs}"], d[f"after_fit_mad_{obs}"]],
                color=["#c9d7ea", "#2a78d6"],
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["raw", "after $\\Delta$"], fontsize=8)
            ax.set_title(f"{name} — {obs}", fontsize=9)
            if c == 0:
                ax.set_ylabel("robust scatter")
    fig.suptitle("Residual scatter about the progenitor orbit, before and after $\\Delta\\Theta(\\phi_1)$")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--T-gyr", type=float, default=0.6, help="initial orbit half-window")
    ap.add_argument("--knot-myr", type=float, default=0.2)
    ap.add_argument("--n-interior-knots", type=int, default=3, help="0 = Ibata's quartic polynomial")
    ap.add_argument("--n-bins", type=int, default=40)
    ap.add_argument("--frame", choices=["streamfinder", "palau"], default="streamfinder")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    agama = _agama()
    tu = agama.getUnits()["time"]
    time_unit_gyr = float(getattr(tu, "value", tu)) / 1e3
    T0 = args.T_gyr / time_unit_gyr
    knot_dt = args.knot_myr * 1e-3 / time_unit_gyr

    sim_path = Path(args.sim)
    data = np.load(sim_path, allow_pickle=False)
    if "sim_data_carthesian" not in data.files:
        raise SystemExit("the fiducial run must NOT use store_window_subsample (no sim_data_carthesian)")
    xv_all = np.asarray(data["sim_data_carthesian"])[args.row]
    j_all = [int(round(v)) for v in np.asarray(data["j"])[args.row, :, 0]]
    pot_cfg = pot_cfg_from_snapshot(sim_path)
    Rs = stream_frames(args.frame)

    out_arrays: dict = {}
    report = {
        "fiducial": str(sim_path),
        "frame": args.frame,
        "T0_gyr": args.T_gyr,
        "knot_myr": args.knot_myr,
        "n_interior_knots": args.n_interior_knots,
        "n_bins": args.n_bins,
        "pot_cfg": {k: str(v) for k, v in pot_cfg.items()},
        "streams": {},
    }

    for s, j in enumerate(j_all):
        name = NAMES.get(j, str(j))
        row = row_of(data, s)
        pot_fid = _host_potential(agama, row, pot_cfg)
        posvel, frame = progenitor_state(agama, pot_fid, row)
        R = Rs[j]

        tpl = oo.build_offset_template(
            agama, pot_fid, posvel, frame, xv_all[s], R,
            T0=T0, knot_dt=knot_dt,
            n_interior_knots=args.n_interior_knots, n_bins=args.n_bins,
        )
        check = self_test(agama, pot_fid, posvel, frame, R, tpl, xv_all[s])
        print(
            f"{name:9s} phi1 [{tpl.phi1_lo:7.2f},{tpl.phi1_hi:7.2f}] "
            f"track_err(phi2) {check['track_err_phi2']:.3f} deg  "
            f"width_ratio(phi2) {check['width_ratio_phi2']:.3f}  "
            f"miss {check['phi1_miss_frac']:.4f}  d(phi1) {check['induced_dphi1_mad_deg']:.3f} deg"
        )

        out_arrays[f"R_{j}"] = tpl.R
        out_arrays[f"phi1_cdf_q_{j}"] = tpl.phi1_cdf_q
        out_arrays[f"phi1_cdf_x_{j}"] = tpl.phi1_cdf_x
        out_arrays[f"phi1_lo_{j}"] = np.float64(tpl.phi1_lo)
        out_arrays[f"phi1_hi_{j}"] = np.float64(tpl.phi1_hi)
        for obs in oo.OBS_NAMES:
            for kind, table in (("delta", tpl.delta), ("sigma", tpl.sigma)):
                sp = table[obs]
                out_arrays[f"{kind}_{obs}_t_{j}"] = np.asarray(sp.t, dtype=float)
                out_arrays[f"{kind}_{obs}_c_{j}"] = np.asarray(sp.c, dtype=float)
            out_arrays[f"sigma_bounds_{obs}_{j}"] = np.asarray(tpl.sigma_bounds[obs], dtype=float)
        for key in FROZEN_LOCAL_KEYS:
            out_arrays[f"fiducial_{key}_{j}"] = np.float64(row[key])
        report["streams"][name] = {"j": j, "diagnostics": tpl.diagnostics, "self_test": check}

    out_arrays["streams"] = np.asarray(sorted(j_all), dtype=np.int64)
    out_arrays["T0"] = np.float64(T0)
    out_arrays["knot_dt"] = np.float64(knot_dt)
    out_arrays["degree"] = np.int64(oo.SPLINE_DEGREE)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **out_arrays)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float))
    fig = Path(args.fig) if args.fig else out.with_name(out.stem + "_diagnostics.png")
    make_figure(report, fig)
    print(f"\nwrote {out}\n      {out.with_suffix('.json')}\n      {fig}")


if __name__ == "__main__":
    main()
