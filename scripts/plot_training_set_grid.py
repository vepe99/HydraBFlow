"""Plot the BINNED network input (``sim_summary`` from ``stream_summary_grid``) for a grouped
sample npz — the values the TimeSeriesTransformer is actually fed — with the real Gaia grid on top.

One figure per stream: 12 panels (10 statistic channels + n_track + n_vlos) vs the phi1 bin centre;
thin coloured lines = sample rows, black = real members. Red crosses mark sim cells whose occupancy
is below ``summary_min_count``: their statistic is a substituted 0, which a plain (unmasked)
TimeSeriesTransformer reads as a measurement.

    .venv/bin/python scripts/plot_training_set_grid.py --sim <grouped npz> --out <prefix> \
        --simulator stream_agama_ou24 [--aug stream_global_ibata_grid] [--real-aug stream_real_global_ibata_grid]
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_grid_coverage import NAMES, OCC, STAT, compose_aug, run_chain  # noqa: E402

UNITS = {"phi2": "deg", "plx": "mas", "mu_phi1": "mas/yr", "mu_phi2": "mas/yr", "vlos": "km/s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="grouped npz (N,S,P,6) + j")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--simulator", required=True)
    ap.add_argument("--aug", default="stream_global_ibata_grid")
    ap.add_argument("--real-aug", default="stream_real_global_ibata_grid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # --- sims through the training chain up to the grid
    aug = compose_aug(args.simulator, args.aug)
    steps = [str(s) for s in aug.steps]
    steps = [s for s in steps[: steps.index("stream_summary_grid") + 1]
             if s not in ("add_noise_to_vcirc", "log10_vcirc")]
    sd = np.load(args.sim)
    x = sd["sim_data_projected"]; n, s_, p, c = x.shape
    jj = np.asarray(sd["j"]).reshape(n * s_).astype(int)
    batch = run_chain(aug, steps, {"sim_data_projected": np.asarray(x, np.float32).reshape(n * s_, p, c),
                                   "j": jj[:, None].astype(np.float32)}, args.seed)
    ssum = np.asarray(batch["sim_summary"])                     # (n*s, K, C)
    min_count = int(aug.params.get("summary_min_count", 3))
    K, C = ssum.shape[1:]
    assert C == 14, f"expected the 14-channel layout, got {C}"

    # --- real members through the real preset's grid step (same frame / edges / estimator)
    raug = compose_aug(args.simulator, args.real_aug)
    d = np.load(args.real)
    m = int(np.asarray(d["j"]).size); max_p = int(aug.params.get("max_particles", 300))
    rb = {}
    for k in d.files:
        if k == "j":
            continue
        a = np.asarray(d[k], dtype=np.float32)
        a = a[:, :max_p] if a.ndim == 2 else a[:, :, :max_p]
        if a.ndim >= 3 and a.shape[0] == 1 and a.shape[1] == m:
            a = a.reshape(m, *a.shape[2:])
        if k in ("attention_mask", "vlos_mask") and a.ndim == 2:
            a = a[:, None, :]
        rb[k] = a
    rb["j"] = np.asarray(d["j"]).reshape(m, 1).astype(np.float32)
    rsum = np.asarray(run_chain(raug, ["stream_summary_grid"], rb, args.seed)["sim_summary"])
    rj = rb["j"].reshape(-1).astype(int)

    labels = STAT + OCC
    cmap = plt.get_cmap("viridis")
    for row in range(m):
        j = int(rj[row]); S = ssum[jj == j]; R = rsum[row]; phi1 = R[:, -1]
        fig, axes = plt.subplots(3, 4, figsize=(17, 10)); axes = axes.ravel()
        for ci, lab in enumerate(labels):
            ax = axes[ci]
            occ_ch = 11 if (lab.endswith("vlos") and lab in STAT) else 10
            for i in range(len(S)):
                ax.plot(S[i, :, -1], S[i, :, ci], "-", lw=0.7, alpha=0.5, color=cmap(i / max(len(S) - 1, 1)))
                if lab in STAT:
                    bad = S[i, :, occ_ch] < min_count
                    ax.plot(S[i, bad, -1], S[i, bad, ci], "x", ms=4, color="r", alpha=0.6)
            ax.plot(phi1, R[:, ci], "k.-", lw=1.8, ms=6, zorder=5, label="real Gaia")
            unit = UNITS.get(lab.split("_", 1)[1], "") if lab in STAT else "stars"
            ax.set_title(f"{lab} [{unit}]", fontsize=10)
            ax.set_xlabel("phi1 bin centre [deg]", fontsize=8)
            if lab in STAT:
                fin = S[:, :, ci][S[:, :, occ_ch] >= min_count]
                lo, hi = np.percentile(np.concatenate([fin, R[:, ci]]), [1, 99]) if fin.size else (R[:, ci].min(), R[:, ci].max())
                pad = 0.15 * (hi - lo + 1e-9); ax.set_ylim(min(lo, 0) - pad, hi + pad)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{NAMES[j]} — binned sim_summary as fed to the TimeSeriesTransformer, {len(S)} training rows "
                     f"(coloured) vs real Gaia (black); red x = occupancy < {min_count} (statistic substituted with 0)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = f"{args.out}_{NAMES[j]}_grid.png"; fig.savefig(out, dpi=120); plt.close(fig); print(f"wrote {out}")
        occ = S[:, :, 10]
        print(f"{NAMES[j]}: cells with n_track<{min_count}: {(occ < min_count).mean():.1%}; "
              f"rows with >=5 such bins: {((occ < min_count).sum(1) >= 5).mean():.1%}; "
              f"n_vlos<{min_count}: {(S[:, :, 11] < min_count).mean():.1%}; "
              f"real n_track per bin {R[:, 10].astype(int).tolist()}, n_vlos {R[:, 11].astype(int).tolist()}")


if __name__ == "__main__":
    main()
