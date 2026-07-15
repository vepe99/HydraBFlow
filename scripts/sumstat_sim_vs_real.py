"""Per-statistic sim-vs-real comparison of the hand-crafted stream summary statistics.

The pooled MMD test (``pipeline/misspecification.py``) and its per-channel variant
(``scripts/misspecification_per_channel.py``) say THAT the observed streams are atypical vs the
simulated reference — this script says WHICH statistic drives it. It rebuilds ``sim_summary``
(the ``stream_summary_grid`` augmentation output: per φ1 bin, [median, std] of the five
stream-frame observables φ2 / parallax / μ_φ1 / μ_φ2 / v_los) for BOTH sides with the exact eval
pipelines — ``_load_test_data`` + ``flatten_members`` + one augmentation draw for the simulated
reference (mirrors evaluate.py), ``_prepare_real_members`` + ``pipeline.transform`` + the real
augmentation chain for the observed Gaia members (mirrors evaluate_real.py) — and then compares
each (stream, statistic, φ1-bin) cell of the real vector against the simulated reference
distribution of the same stream: robust z-score (vs median/1.4826·MAD) and percentile rank.

Because ``sim_summary`` is computed by the augmentation BEFORE any network, the result is
model-free (identical for every trial trained on this dataset): it localizes the
misspecification in interpretable data space (degrees, mas, mas/yr, km/s).

Empty-bin caveat: the augmentation fills empty bins with exactly 0.0 (``nan_to_num``). v_los bins
(measured stars only) are often empty in sims; cells where the real value is exactly 0.0 or where
most of the sim pool is 0.0 are masked from the z map and flagged in the JSON.

CPU only (forces JAX_PLATFORMS=cpu). Usage:
    uv run python scripts/sumstat_sim_vs_real.py \
        --sim-run  outputs/ibata_onedisk_grid_m200c/tuning/best_trial36/eval_sim \
        --real-run outputs/ibata_onedisk_grid_m200c/tuning/best_trial36/eval_real
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

# stream_summary_grid channel layout: 5 observables x [median, std], then j, phi1_centre.
_OBS = ["phi2", "parallax", "mu_phi1", "mu_phi2", "vlos"]
_UNITS = {"phi2": "deg", "parallax": "mas", "mu_phi1": "mas/yr", "mu_phi2": "mas/yr",
          "vlos": "km/s"}
_STATS = [f"{s}_{o}" for o in _OBS for s in ("med", "std")]  # channel c = 2*obs + (0 med, 1 std)


def _stat_channels():
    return {f"{s}_{o}": 2 * i + (0 if s == "med" else 1)
            for i, o in enumerate(_OBS) for s in ("med", "std")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-run", required=True, help="evaluate composition=global run dir (reference)")
    ap.add_argument("--real-run", required=True, help="evaluate_real composition=global run dir (observed)")
    ap.add_argument("--model-dir", default=None, help="train dir (default: sim-run's model_dir)")
    ap.add_argument("--zero-frac-mask", type=float, default=0.5,
                    help="mask cells where more than this fraction of the sim pool is exactly 0")
    ap.add_argument("--out-dir", default=None, help="artifact dir (default: real-run)")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from hydrabflow.pipeline.adapter import (
        fill_adapter_from_simulator, fill_stream_grid_from_simulator,
    )
    from hydrabflow.pipeline.compositional import apply_augmentations_once, flatten_members
    from hydrabflow.pipeline.evaluate import _load_test_data
    from hydrabflow.pipeline.evaluate_real import _prepare_real_members
    from hydrabflow.preprocessing.registry import build_pipeline
    from hydrabflow.utils.paths import PREPROCESSING_STATE

    def load_cfg(run):
        c = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml"))
        OmegaConf.set_struct(c, False)
        fill_adapter_from_simulator(c)
        fill_stream_grid_from_simulator(c)
        return c

    cs = load_cfg(args.sim_run)
    cr = load_cfg(args.real_run)
    md = args.model_dir or cs.get("model_dir")

    # Simulated reference: flat member rows + one augmentation draw (mirrors evaluate.py).
    test_data, pipeline = _load_test_data(cs, md)
    ctx = next(iter(cs.adapter.inference_conditions), "j")
    n, m = np.asarray(test_data[ctx]).shape[:2]
    flat_sim = apply_augmentations_once(flatten_members(test_data, m), cs, pipeline, int(cs.seed))

    # Observed Gaia members (mirrors evaluate_real.py).
    real_pipeline = build_pipeline(cr.preprocessing)
    real_pipeline.load(os.path.join(md, PREPROCESSING_STATE))
    flat_real, mr = _prepare_real_members(cr)
    flat_real = real_pipeline.transform(flat_real)
    flat_real = apply_augmentations_once(flat_real, cr, real_pipeline, int(cr.seed))

    S_sim = np.asarray(flat_sim["sim_summary"], float)   # (n*m, K, 12)
    S_real = np.asarray(flat_real["sim_summary"], float)  # (mr, K, 12)
    j_sim = np.asarray(flat_sim["j"]).reshape(-1).astype(int)
    j_real = np.asarray(flat_real["j"]).reshape(-1).astype(int)
    K = S_sim.shape[1]
    chan = _stat_channels()

    try:
        from hydrabflow.simulators.registry import get_simulator
        streams = getattr(get_simulator(cs.simulator), "target_streams", None) or {}
        jname = {int(v): str(k) for k, v in streams.items()}
    except Exception:
        jname = {}

    print(f"sim_summary: sim {S_sim.shape} vs real {S_real.shape} | K={K} phi1 bins x "
          f"{len(_STATS)} statistics")

    # ---- per-cell robust z + percentile -----------------------------------------------------
    results = {}
    zmaps = {}  # stream name -> (n_stats, K) masked z array
    for i, j in enumerate(j_real):
        name = jname.get(j, f"member_{j}")
        pool = S_sim[j_sim == j]                       # (N_j, K, 12)
        centres = np.median(pool[..., -1], axis=0)     # phi1 bin centres (fixed per stream)
        z = np.full((len(_STATS), K), np.nan)
        pct = np.full((len(_STATS), K), np.nan)
        cells = {}
        for si, stat in enumerate(_STATS):
            c = chan[stat]
            sim_v = pool[:, :, c]                      # (N_j, K)
            real_v = S_real[i, :, c]                   # (K,)
            zero_frac = (sim_v == 0.0).mean(axis=0)
            med = np.median(sim_v, axis=0)
            mad = 1.4826 * np.median(np.abs(sim_v - med), axis=0) + 1e-12
            zz = (real_v - med) / mad
            pp = np.array([(sim_v[:, k] <= real_v[k]).mean() * 100 for k in range(K)])
            bad = (zero_frac > args.zero_frac_mask) | (real_v == 0.0)
            zz[bad], pp[bad] = np.nan, np.nan
            z[si], pct[si] = zz, pp
            cells[stat] = {
                "z": [None if not np.isfinite(v) else round(float(v), 2) for v in zz],
                "percentile": [None if not np.isfinite(v) else round(float(v), 1) for v in pp],
                "sim_zero_frac": [round(float(v), 2) for v in zero_frac],
            }
        zmaps[name] = z
        results[name] = {"phi1_centres_deg": [round(float(v), 2) for v in centres],
                         "n_reference": int(pool.shape[0]), "cells": cells}

    # ---- worst offenders --------------------------------------------------------------------
    print("\nWorst cells (|robust z| ranked, top 15 per stream):")
    offenders = {}
    for name, z in zmaps.items():
        flatz = [(abs(z[si, k]), _STATS[si], k, z[si, k])
                 for si in range(len(_STATS)) for k in range(K) if np.isfinite(z[si, k])]
        flatz.sort(reverse=True)
        offenders[name] = [{"stat": s, "bin": k, "z": round(v, 2)} for _, s, k, v in flatz[:15]]
        print(f"  {name}:")
        for o in offenders[name][:8]:
            print(f"    {o['stat']:14s} bin {o['bin']:2d}  z={o['z']:+6.2f}")

    # per-statistic aggregate (median |z| over bins+streams): which OBSERVABLE is off overall
    agg = {}
    for si, stat in enumerate(_STATS):
        vals = np.concatenate([z[si][np.isfinite(z[si])] for z in zmaps.values()])
        if vals.size:
            agg[stat] = {"median_abs_z": round(float(np.median(np.abs(vals))), 2),
                         "max_abs_z": round(float(np.max(np.abs(vals))), 2),
                         "n_cells": int(vals.size)}
    print("\nPer-statistic aggregate over all streams/bins (median |z| / max |z|):")
    for stat, a in sorted(agg.items(), key=lambda kv: -kv[1]["median_abs_z"]):
        print(f"  {stat:14s} {a['median_abs_z']:5.2f} / {a['max_abs_z']:6.2f}  ({a['n_cells']} cells)")

    # ---- figures ----------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.out_dir or args.real_run
    names = list(zmaps)

    # (1) z heatmap: one panel per stream, rows=statistics, cols=phi1 bins
    fig, axes = plt.subplots(1, len(names), figsize=(0.62 * K * len(names) + 3, 5.4),
                             squeeze=False)
    axes = axes[0]
    vmax = 8.0
    for ax, name in zip(axes, names):
        im = ax.imshow(np.clip(zmaps[name], -vmax, vmax), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_title(name)
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"{v:.0f}" for v in results[name]["phi1_centres_deg"]], fontsize=6,
                           rotation=45)
        ax.set_xlabel(r"$\phi_1$ bin centre [deg]")
        if ax is axes[0]:
            ax.set_yticks(range(len(_STATS)))
            ax.set_yticklabels(_STATS, fontsize=7)
        else:
            ax.set_yticks([])
        for si in range(len(_STATS)):
            for k in range(K):
                v = zmaps[name][si, k]
                if np.isfinite(v) and abs(v) >= 3:
                    ax.text(k, si, f"{v:+.0f}", ha="center", va="center", fontsize=5.5)
                elif not np.isfinite(v):
                    ax.text(k, si, "x", ha="center", va="center", fontsize=5, color="0.6")
    fig.colorbar(im, ax=axes, shrink=0.85, label="robust z (real vs sim reference)")
    fig.suptitle("Per-statistic sim-vs-real z-scores of the stream summary statistics "
                 "(x = masked empty bin; |z|>=3 annotated)", y=1.02)
    f1 = os.path.join(out_dir, "sumstat_sim_vs_real_zmap.png")
    fig.savefig(f1, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # (2) track overlays: per stream, 2 rows (median, std) x 5 observables vs phi1
    colors = {"Pal5": "#d7191c", "NGC3201": "#2c7bb6", "M68": "#fdae61"}
    fig, axes = plt.subplots(2 * len(names), len(_OBS),
                             figsize=(3.5 * len(_OBS), 2.6 * 2 * len(names)), squeeze=False)
    for gi, name in enumerate(names):
        j = [k for k, v in jname.items() if v == name]
        j = j[0] if j else int(j_real[gi])
        pool = S_sim[j_sim == j]
        x = results[name]["phi1_centres_deg"]
        i_real = int(np.flatnonzero(j_real == j)[0])
        for oi, obs in enumerate(_OBS):
            for ri, statname in enumerate(("med", "std")):
                ax = axes[2 * gi + ri][oi]
                c = chan[f"{statname}_{obs}"]
                sim_v = pool[:, :, c].astype(float)
                sim_v[sim_v == 0.0] = np.nan  # empty-bin marker
                lo95, lo68, med, hi68, hi95 = np.nanpercentile(sim_v, [2.5, 16, 50, 84, 97.5],
                                                               axis=0)
                col = colors.get(name, "purple")
                ax.fill_between(x, lo95, hi95, color=col, alpha=0.15, lw=0)
                ax.fill_between(x, lo68, hi68, color=col, alpha=0.30, lw=0)
                ax.plot(x, med, color=col, lw=1.4, label="sim reference")
                rv = S_real[i_real, :, c].astype(float)
                rv[rv == 0.0] = np.nan
                ax.plot(x, rv, "ko-", ms=3.5, lw=1.2, label="real Gaia", zorder=5)
                if ri == 0 and gi == 0:
                    ax.set_title(obs)
                if oi == 0:
                    ax.set_ylabel(f"{name}\n{statname} [{_UNITS[obs]}]", fontsize=8)
                if 2 * gi + ri == 2 * len(names) - 1:
                    ax.set_xlabel(r"$\phi_1$ [deg]")
                ax.grid(alpha=0.2)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Stream-frame summary tracks: real Gaia vs simulated reference "
                 "(bands = 68/95% of sim members)", y=1.005)
    fig.tight_layout()
    f2 = os.path.join(out_dir, "sumstat_sim_vs_real_tracks.png")
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload = {"sim_run": args.sim_run, "real_run": args.real_run,
               "per_statistic_aggregate": agg, "worst_cells": offenders,
               "figures": [f1, f2], "streams": results}
    with open(os.path.join(out_dir, "sumstat_sim_vs_real.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {f1}\nSaved {f2}")


if __name__ == "__main__":
    main()
