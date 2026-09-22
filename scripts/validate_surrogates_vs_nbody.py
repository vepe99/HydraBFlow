#!/usr/bin/env python
"""Measure the surrogate forward models against restricted-N-body truth, across the prior.

Both surrogates in this project replay ONE stripping history: ``stream_trihedron`` freezes stars
into the Frenet-Serret trihedron of the fiducial orbit, ``stream_orbit_offset`` fits Ibata+2024
DeltaTheta(phi1) corrections to it. Both are exact-ish at their own fiducial and degrade away from
it, and the 2026-08-28 sweeps showed the trihedron losing badly along the halo-MASS axis (a mass
change is mostly an orbital-period rescaling, which is exactly the along-track redistribution a
frozen template cannot do). That bias was never measured over the training prior, which is what
this script does.

At each point of a (log10 M200, q_halo) grid -- every other global at its prior centre, every local
at the template's fiducial -- all three forward models run on IDENTICAL parameters, and the
surrogates are scored against the N-body cloud in the published STREAMFINDER frames:

  track    median over phi1 bins of |median phi2 sim - median phi2 truth|, deg
  width    median over phi1 bins of the robust phi2 dispersion ratio sim/truth (1 = right)
  edge     phi1 edge/centre number-density ratio, sim/truth. This is the statistic that exposes a
           frozen along-track density, and it is the one both surrogates are expected to fail.
  n_ratio  in-window star count sim/truth -- a surrogate can be on-track and still put the wrong
           number of stars in the observation window.
  mmd      RBF MMD^2 between the two clouds over (phi1, phi2, parallax, mu_phi1, mu_phi2, v_los),
           standardized on the truth cloud.

Scored on the NOISE-FREE in-window stars: the Gaia observation model would wash out exactly the
differences this is trying to measure.

    .venv/bin/python scripts/validate_surrogates_vs_nbody.py --n-grid 5 --out outputs/surrogate_validation
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

from hydra import compose, initialize_config_dir  # noqa: E402

from hydrabflow.config import register_configs  # noqa: E402
from hydrabflow.registry import get_simulator  # noqa: E402
from hydrabflow.simulators import stream_frame as sf  # noqa: E402
from streamfinder_frame import frames as streamfinder_frames  # noqa: E402

NAMES = {0: "Pal5", 1: "NGC3201", 2: "M68"}
TRUTH = "stream_agama_rnbody_ibata_m200c_v4"
SURROGATES = {
    "orbit_offset": "stream_orbit_offset_mcmillan17_v4",
    "trihedron": "stream_trihedron_mcmillan17_v4",
}
MASS_KEY = "log10_M200_TwoPowerTriaxial_halo"
Q_KEY = "q_TwoPowerTriaxial_halo"
PAD = -999.0


def load_sim(name: str, n_workers: int):
    register_configs()
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"simulator={name}", "composition=global",
                       f"++simulator.params.n_workers={n_workers}"],
        )
    return get_simulator(cfg.simulator)


def centre(spec) -> float:
    """The centre of a prior: mean for normal, midpoint for uniform, the value for identity."""
    kind, pp = str(spec["type"]), [float(v) for v in spec["prior_parameters"]]
    if kind == "uniform":
        return 0.5 * (pp[0] + pp[1])
    return pp[0]


def grid_params(truth_sim, surrogate_sim, masses, qs) -> dict:
    """One row per (grid point, stream), identical for every arm.

    Globals at their prior centre except the two swept; locals at the SURROGATE's pinned values, so
    the N-body arm runs the very stripping history the templates were built from.
    """
    streams = list(truth_sim.target_streams.items())
    rows = [(m, q, name, j) for m in masses for q in qs for name, j in streams]
    n = len(rows)

    out: dict[str, np.ndarray] = {}
    for key, spec in truth_sim._priors_global.items():
        out[key] = np.full((n, 1), centre(spec))
    out[MASS_KEY] = np.array([[m] for m, _, _, _ in rows])
    out[Q_KEY] = np.array([[q] for _, q, _, _ in rows])

    local_keys = list(next(iter(truth_sim._priors_local.values())).keys())
    for key in local_keys:
        col = np.empty((n, 1))
        for i, (_, _, name, _) in enumerate(rows):
            spec = surrogate_sim._priors_local[name].get(key, truth_sim._priors_local[name][key])
            col[i, 0] = centre(spec)
        out[key] = col
    out["j"] = np.array([[float(j)] for _, _, _, j in rows])
    return out, rows


def clouds(stars: np.ndarray, j: int, R_by_j: dict):
    """In-window, non-padded stars of one row, in its published stream frame."""
    a = np.asarray(stars, dtype=float)
    a = a[(a[:, 0] != PAD) & np.isfinite(a).all(axis=1)]
    if a.shape[0] < 20:
        return None
    phi1, phi2, m1, m2 = sf.project(R_by_j[j], a[:, 0], a[:, 1], a[:, 3], a[:, 4])
    phi1 = sf.unwrap_to(phi1, np.full(phi1.shape, np.median(phi1)))
    return np.column_stack([phi1, phi2, 1.0 / a[:, 2], m1, m2, a[:, 5]])


def mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def edge_centre(phi1, edges):
    """Number-density ratio of the outer phi1 quintiles to the central three."""
    lo, hi = edges[0], edges[-1]
    w = (hi - lo) / 5.0
    outer = ((phi1 < lo + w) | (phi1 > hi - w)).sum() / (2 * w)
    inner = ((phi1 >= lo + w) & (phi1 <= hi - w)).sum() / (3 * w)
    return float(outer / inner) if inner > 0 else float("nan")


def mmd2(X, Y, gamma):
    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)

    n, m = len(X), len(Y)
    kxx, kyy = k(X, X), k(Y, Y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * k(X, Y).mean())


def score(truth, surro, n_bins=8, n_mmd=400, seed=0):
    rng = np.random.default_rng(seed)
    edges = np.quantile(truth[:, 0], np.linspace(0, 1, n_bins + 1))
    it = np.clip(np.digitize(truth[:, 0], edges) - 1, 0, n_bins - 1)
    isx = np.clip(np.digitize(surro[:, 0], edges) - 1, 0, n_bins - 1)

    terr, wrat = [], []
    for b in range(n_bins):
        a, c = truth[it == b, 1], surro[isx == b, 1]
        if len(a) < 5 or len(c) < 5:
            continue
        terr.append(abs(np.median(c) - np.median(a)))
        st = mad(a)
        wrat.append(mad(c) / st if st > 0 else np.nan)

    mu, sc = np.median(truth, 0), np.array([mad(truth[:, c]) for c in range(6)])
    sc[sc <= 0] = 1.0
    A = (truth[rng.choice(len(truth), min(n_mmd, len(truth)), replace=False)] - mu) / sc
    B = (surro[rng.choice(len(surro), min(n_mmd, len(surro)), replace=False)] - mu) / sc
    d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    return {
        "track": float(np.nanmedian(terr)) if terr else float("nan"),
        "width": float(np.nanmedian(wrat)) if wrat else float("nan"),
        "edge": float(edge_centre(surro[:, 0], edges) / edge_centre(truth[:, 0], edges)),
        "n_ratio": float(len(surro) / len(truth)),
        "mmd": mmd2(A, B, 1.0 / (2.0 * med)),
    }


def figure(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["track", "width", "edge", "mmd"]
    arms = list(SURROGATES)
    colours = {"orbit_offset": "#2a78d6", "trihedron": "#eb6834"}
    fig, axes = plt.subplots(len(metrics), 6, figsize=(21, 3.0 * len(metrics)), squeeze=False)
    rows = report["rows"]
    for r, met in enumerate(metrics):
        for c, (axis, label) in enumerate([(MASS_KEY, "log10 M200"), (Q_KEY, "q halo")]):
            for s, sname in enumerate(NAMES.values()):
                ax = axes[r][c * 3 + s]
                for arm in arms:
                    pts = [(d[axis], d[arm][met]) for d in rows if d["stream"] == sname and arm in d]
                    if not pts:
                        continue
                    x = np.array([p[0] for p in pts])
                    y = np.array([p[1] for p in pts])
                    o = np.argsort(x)
                    ax.plot(x[o], y[o], "o", ms=3.5, alpha=0.75, color=colours[arm], label=arm)
                if met in ("width", "edge", "n_ratio"):
                    ax.axhline(1.0, color="#52514e", lw=0.8, ls="--")
                ax.set_yscale("log" if met in ("mmd",) else "linear")
                if r == 0:
                    ax.set_title(f"{sname} — vs {label}", fontsize=9)
                if c * 3 + s == 0:
                    ax.set_ylabel(met)
                if r == len(metrics) - 1:
                    ax.set_xlabel(label)
                if r == 0 and c * 3 + s == 0:
                    ax.legend(fontsize=7)
    fig.suptitle("Surrogate forward models vs restricted-N-body truth, over the v4 prior")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-grid", type=int, default=5)
    ap.add_argument("--n-workers", type=int, default=24)
    ap.add_argument("--out", default="outputs/surrogate_validation")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    truth_sim = load_sim(TRUTH, args.n_workers)
    sims = {k: load_sim(v, args.n_workers) for k, v in SURROGATES.items()}
    ref = sims["orbit_offset"]

    mspec = truth_sim._priors_global[MASS_KEY]["prior_parameters"]
    qspec = truth_sim._priors_global[Q_KEY]["prior_parameters"]
    masses = np.linspace(float(mspec[0]), float(mspec[1]), args.n_grid)
    qs = np.linspace(float(qspec[0]), float(qspec[1]), args.n_grid)
    params, rows = grid_params(truth_sim, ref, masses, qs)
    print(f"{len(rows)} rows per arm ({args.n_grid}x{args.n_grid} grid x {len(NAMES)} streams)")

    R_by_j = {j: R for j, R in
              zip(NAMES, [streamfinder_frames()[n] for n in NAMES.values()])}

    stars = {}
    for arm, sim in [("truth", truth_sim)] + list(sims.items()):
        print(f"  simulating {arm} ...", flush=True)
        stars[arm] = sim.simulate(params, np.random.default_rng(args.seed))["sim_data_projected"]

    report = {"grid": {"masses": masses.tolist(), "qs": qs.tolist()}, "rows": []}
    for i, (m, q, name, j) in enumerate(rows):
        t = clouds(stars["truth"][i], j, R_by_j)
        rec = {"stream": name, MASS_KEY: float(m), Q_KEY: float(q)}
        if t is None:
            rec["skipped"] = "N-body truth has too few in-window stars"
            report["rows"].append(rec)
            continue
        for arm in sims:
            s = clouds(stars[arm][i], j, R_by_j)
            if s is not None:
                rec[arm] = score(t, s)
        report["rows"].append(rec)

    summary = {}
    for arm in sims:
        summary[arm] = {}
        for name in NAMES.values():
            vals = [d[arm] for d in report["rows"] if d.get("stream") == name and arm in d]
            summary[arm][name] = {
                met: float(np.nanmedian([v[met] for v in vals])) if vals else float("nan")
                for met in ("track", "width", "edge", "n_ratio", "mmd")
            }
    report["summary"] = summary

    (out / "validation_vs_nbody.json").write_text(json.dumps(report, indent=2, default=float))
    figure(report, out / "validation_vs_nbody.png")

    print("\nmedian over the grid (track deg | width | edge | n_ratio | mmd)")
    for arm in sims:
        print(f"  {arm}")
        for name in NAMES.values():
            s = summary[arm][name]
            print(
                f"    {name:9s} {s['track']:7.3f} | {s['width']:6.3f} | {s['edge']:6.3f} "
                f"| {s['n_ratio']:6.3f} | {s['mmd']:.4f}"
            )
    print(f"\nwrote {out/'validation_vs_nbody.json'}\n      {out/'validation_vs_nbody.png'}")


if __name__ == "__main__":
    main()
