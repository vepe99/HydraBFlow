"""Referee point 1 figure: masked compositional plan (m stream items + 1 curve item, the curve counted
once) vs the naive plan (every item = stream_j + curve, the curve counted m times). Same model, test
set, seed and bridge d1; the naive plan is shown ONLY to illustrate the over-counting.

Panels: rank-ECDF calibration of each plan on the simulated test set, and the per-dataset ratio of
posterior widths (naive / masked) per parameter, with the sqrt(m) bound; the real-Gaia ratio is
overlaid when both real posteriors are given.

    python scripts/plot_plan_comparison.py --masked <dir> --naive <dir> --truths <eval_sim>/truths_flat.npz \
        [--masked-real <posterior.npz> --naive-real <posterior.npz>] --out <dir>
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_calibration_ecdf import LABEL, draw_ecdf, save  # noqa: E402

POST = "compositional_posterior.npz"


def std(d, k):
    return np.asarray(d[k])[..., 0].std(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--masked", required=True)
    ap.add_argument("--naive", required=True)
    ap.add_argument("--truths", required=True)
    ap.add_argument("--masked-real")
    ap.add_argument("--naive-real")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

    masked, naive = (dict(np.load(os.path.join(d, POST))) for d in (args.masked, args.naive))
    keys = list(masked)
    tr = np.load(args.truths)
    assert (tr["j"].reshape(-1, args.m) == np.arange(args.m)).all()  # group-major member rows
    targ = {k: tr[k][:: args.m] for k in keys}  # globals are repeated per member
    assert len(targ[keys[0]]) == masked[keys[0]].shape[0]

    ratio = {k: std(naive, k) / std(masked, k) for k in keys}
    real = None
    if args.masked_real and args.naive_real:
        mr, nr = np.load(args.masked_real), np.load(args.naive_real)
        real = {k: float(std(nr, k)[0] / std(mr, k)[0]) for k in keys}

    def calib(est):
        return {k: float(np.median(np.abs(np.sort(np.mean(est[k][..., 0] < targ[k][:, None], axis=1))
                                          - (np.arange(len(targ[k])) + 0.5) / len(targ[k]))))
                for k in keys}

    summary = {"width_ratio_median": {k: float(np.median(v)) for k, v in ratio.items()},
               "width_ratio_real": real, "calib_masked": calib(masked), "calib_naive": calib(naive)}
    for name, d in (("masked", args.masked), ("naive", args.naive)):
        with open(os.path.join(d, "compositional_metrics.json")) as f:
            summary[f"metrics_{name}"] = json.load(f)
    with open(os.path.join(args.out, "plan_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("width_ratio_median", "width_ratio_real")}, indent=1))

    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, (est, title) in zip(axs, [(masked, f"{args.m} stream items + 1 curve item"),
                                      (naive, f"{args.m} items of (stream + curve)")]):
        draw_ecdf(ax, est, targ, keys)
        ax.set_title(title, fontsize=11)
    lim = max(abs(np.array(ax.get_ylim())).max() for ax in axs[:2])
    for ax in axs[:2]:
        ax.set_ylim(-lim, lim)
    axs[1].set_ylabel("")
    axs[0].legend(fontsize=8, frameon=False, loc="lower left", ncol=2)

    ax = axs[2]
    x = np.arange(len(keys))
    ax.boxplot([ratio[k] for k in keys], positions=x, widths=0.55, showfliers=False,
               medianprops=dict(color="k"))
    if real:
        ax.plot(x, [real[k] for k in keys], "D", color=plt.cm.RdYlBu_r(0.0), ms=6, label="real Gaia")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.axhline(1 / np.sqrt(args.m), color="0.5", lw=0.8, ls="--")
    ax.text(len(keys) - 0.5, 1 / np.sqrt(args.m), rf"$1/\sqrt{{{args.m}}}$", va="bottom", ha="right",
            fontsize=9, color="0.4")
    ax.set_xticks(x, [LABEL.get(k, k) for k in keys])
    ax.set_ylabel(r"posterior width ratio  $\sigma_{\rm naive}/\sigma_{\rm masked}$")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, args.out, "plan_comparison")


if __name__ == "__main__":
    main()
