"""Where in the summary PCA basis does the real disk's distance actually come from?

Reads the cached `summaries.npz`. Mahalanobis distance decomposes exactly over PCA components as
sum_k z_k^2 / var_k, so the per-component contribution says whether an out-of-distribution verdict
rests on well-sampled directions or on the low-variance tail -- where the training covariance is
worst estimated and the network may just be carrying noise.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# `check_summary_range.py <plot_dir>` writes the `summaries.npz` this reads, one per model.
# Usage: check_summary_pca.py <out.png> <label>=<plot_dir> [<label>=<plot_dir> ...]
RUNS = [tuple(a.split("=", 1)) for a in sys.argv[2:]]
DISK = os.environ.get("CHECK_DISK", "oph163131")
if not RUNS:
    sys.exit(__doc__ + "\nUsage: check_summary_pca.py <out.png> <label>=<plot_dir> [...]")


def load(d):
    z = np.load(d + "/summaries.npz")
    F, r = z["train"].astype(np.float64), z["real"].astype(np.float64)
    keys, absent = list(z["branch_keys"]), set(z["absent"].tolist())
    w = F.shape[1] // len(keys)
    cols = np.concatenate([np.arange(i * w, (i + 1) * w)
                           for i, k in enumerate(keys) if k not in absent])
    return F[:, cols], r[cols]


def pct(train, value):
    t = np.sort(train)
    return 100.0 * np.searchsorted(t, value) / len(t)


fig, axes = plt.subplots(len(RUNS), 3, figsize=(16.5, 4.2 * len(RUNS)), squeeze=False)
for row, (label, d) in enumerate(RUNS):
    P, r = load(d)
    mu = P.mean(0)
    _u, sv, vt = np.linalg.svd(P - mu, full_matrices=False)
    var = sv ** 2 / (len(P) - 1)
    cum = np.cumsum(var) / var.sum()
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    S = (P - mu) @ vt.T            # (N, 64) training scores
    s = (r - mu) @ vt.T            # (64,)   the disk's scores
    z = s / S.std(0)               # per-component standard score

    # Mahalanobis in the PCA basis is exactly sum z_k^2, and it decomposes per component.
    d2_tr_k = (S / S.std(0)) ** 2
    contrib = z ** 2
    d2_full = contrib.sum()
    d2_99 = contrib[:k99].sum()
    d2_tr_full = d2_tr_k.sum(1)
    d2_tr_99 = d2_tr_k[:, :k99].sum(1)
    print(f"\n=== {label}   {P.shape[1]} dims, {len(P)} training rows")
    print(f"  components for 99% of variance: {k99}  (PC1 {100*var[0]/var.sum():.1f}%, "
          f"PC1-2 {100*cum[1]:.1f}%, PC1-{k99} {100*cum[k99-1]:.2f}%)")
    print(f"  d2 over all {P.shape[1]} PCs : {d2_full:8.1f}   p{pct(d2_tr_full, d2_full):6.2f}")
    print(f"  d2 over the first {k99:2d} PCs : {d2_99:8.1f}   p{pct(d2_tr_99, d2_99):6.2f}"
          f"   ({100*d2_99/d2_full:.1f}% of the disk's total d2)")
    print(f"  d2 in the {P.shape[1]-k99} discarded low-variance PCs: {d2_full-d2_99:.1f} "
          f"({100*(1-d2_99/d2_full):.1f}%)")
    top = np.argsort(-contrib)[:6]
    print("  components contributing most:")
    for t in top:
        print(f"    PC{t+1:<3d} var {100*var[t]/var.sum():5.2f}%  z={z[t]:+6.2f}  "
              f"percentile {pct(S[:, t], s[t]):6.2f}  d2 share {100*contrib[t]/d2_full:5.1f}%")
    n_out = int((np.abs(z[:k99]) > 2.576).sum())
    print(f"  of the first {k99} PCs, {n_out} put the disk outside the central 99% "
          f"({100*n_out/k99:.0f}%; 1% expected by chance)")

    a = axes[row][0]
    a.plot(np.arange(1, len(cum) + 1), 100 * cum, color="#0072B2", lw=1.6)
    a.axhline(99, color="0.6", ls="--", lw=1)
    a.axvline(k99, color="#D55E00", lw=1.4)
    a.annotate(f"99% at PC{k99}", (k99, 60), xytext=(6, 0), textcoords="offset points",
               fontsize=9, color="#D55E00")
    a.set_xlabel("component"); a.set_ylabel("cumulative variance (%)")
    a.set_title(f"{label}: {k99} of {P.shape[1]} PCs reach 99%", fontsize=10)

    a = axes[row][1]
    cols = ["#D55E00" if abs(v) > 2.576 else "#0072B2" for v in z]
    a.bar(np.arange(1, len(z) + 1), z, color=cols, width=.85)
    a.axhspan(-2.576, 2.576, color="0.9", zorder=-5)
    a.axvline(k99 + .5, color="0.4", ls=":", lw=1.2)
    a.set_xlabel("component"); a.set_ylabel("disk's score (sd of training)")
    a.set_title(f"{label}: position per component (dotted = 99% cut)", fontsize=10)

    a = axes[row][2]
    a.plot(np.arange(1, len(z) + 1), 100 * np.cumsum(contrib) / d2_full, color="#0072B2", lw=1.6)
    a.axvline(k99 + .5, color="#D55E00", lw=1.4)
    a.axhline(100 * d2_99 / d2_full, color="0.6", ls="--", lw=1)
    a.set_xlabel("component"); a.set_ylabel("cumulative share of the disk's $d^2$ (%)")
    a.set_title(f"{label}: {100*d2_99/d2_full:.0f}% of $d^2$ inside the 99%-variance subspace",
                fontsize=10)

fig.suptitle(f"Summary-space PCA in depth — {DISK} vs the training population", fontsize=12)
fig.tight_layout()
out = sys.argv[1]
fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="white")
print(f"\nwrote {out}")
