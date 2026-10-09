"""Gain from pooling: posterior width of each single-modality posterior divided by the width of the
compositional (pooled) one, per parameter. Ratio > 1 = the combination tightens that parameter.

Simulated test set: box per stream of sigma_stream-only / sigma_pooled over the datasets (stream-only =
`eval.observed_groups=[sim_summary]` base posteriors, group-major rows with j = 0..m-1). Real Gaia:
markers for each stream-only row and the curve-only posterior.

    python scripts/plot_pooling_gain.py --sim-single <eval_sim_only_sim_summary>/base_posterior.npz \
        --sim-pooled <eval_sim>/compositional_posterior.npz --truths <eval_sim>/truths_flat.npz \
        --real-streams <only_sim_summary>/single_stream_posterior.npz \
        --real-curve <only_vcirc_kms>/single_stream_posterior.npz --real-pooled <eval_real>/posterior.npz --out <dir>
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_calibration_ecdf import LABEL, save  # noqa: E402

STREAMS = ["Pal5", "NGC3201", "M68"]


def std(d, k):
    return np.asarray(d[k])[..., 0].std(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for a in ("--sim-single", "--sim-pooled", "--truths", "--real-streams", "--real-curve", "--real-pooled",
              "--out"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

    single, pooled = np.load(args.sim_single), np.load(args.sim_pooled)
    keys = list(pooled.files)
    m = len(STREAMS)
    assert (np.load(args.truths)["j"].reshape(-1, m) == np.arange(m)).all()  # group-major member rows
    n = pooled[keys[0]].shape[0]
    sim = {k: std(single, k).reshape(n, m) / std(pooled, k)[:, None] for k in keys}  # (n, m)

    rs, rc, rp = (np.load(f) for f in (args.real_streams, args.real_curve, args.real_pooled))
    real = {k: {**{s: float(std(rs, k)[i] / std(rp, k)[0]) for i, s in enumerate(STREAMS)},
                "curve": float(std(rc, k)[0] / std(rp, k)[0])} for k in keys}
    summary = {"sim_median": {k: dict(zip(STREAMS, np.median(sim[k], axis=0).tolist())) for k in keys},
               "real": real}
    with open(os.path.join(args.out, "pooling_gain.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))

    colors = dict(zip(STREAMS, plt.cm.RdYlBu_r(np.linspace(0, 1, 4))[1:]))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(keys))
    off = (np.arange(m) - (m - 1) / 2) * 0.24
    for i, s in enumerate(STREAMS):
        bp = ax.boxplot([sim[k][:, i] for k in keys], positions=x + off[i], widths=0.2, showfliers=False,
                        patch_artist=True, medianprops=dict(color="k"))
        for b in bp["boxes"]:
            b.set(facecolor=colors[s], alpha=0.45, edgecolor=colors[s])
        ax.plot(x + off[i], [real[k][s] for k in keys], "D", color=colors[s], mec="k", ms=6, label=f"{s} only")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x, [LABEL.get(k, k) for k in keys])
    ax.set_ylabel(r"$\sigma_{\rm single}/\sigma_{\rm pooled}$")
    ax.set_title("boxes: simulated test set (stream only); markers: real Gaia", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, args.out, "pooling_gain")


if __name__ == "__main__":
    main()
