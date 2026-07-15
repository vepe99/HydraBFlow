"""Per-channel misspecification for the POTENTIAL observables (vcirc, vterm, sigma_z) against a
large training-set reference, noise-free and per-potential.

Motivation: the summary-space MMD test uses the 333-group test set (~333 potentials) as the
reference and runs on the noise-augmented rows, which (a) gives a thin reference cloud and (b) lets
the observational-noise augmentation wash out systematic offsets in the group-level potential
channels. This script instead:
  * uses N (default 10^4) rows of the FLAT training set as the reference -- each flat row is an
    INDEPENDENT potential, so the reference is ~N distinct potentials, not 333;
  * compares the stored NOISE-FREE model curves per potential (no augmentation), which is the
    correct per-potential test for the potential-derived observables.

For each channel it reports the observed MW curve's Mahalanobis percentile within the reference
(full-covariance, ridge-regularized), the per-bin z-scores, and an RBF-MMD (median-heuristic
bandwidth) with a single-draw bootstrap null (observed = 1 potential vs the reference cloud).

Standalone: numpy only. Observed vterm from assets/terminal_velocity.csv; rotation curve inlined
(Zhou 2023 u Huang 2016, same as scripts/ppc_rotation_curve.py); sigma_z datum 71 Msun/pc^2. So it
does NOT import hydrabflow (avoids the auto-discovery -> BayesFlow/JAX import stall) and runs on CPU
in seconds.

Usage:
    python scripts/mmd_potential_vs_training.py \
        --train data/data_jarvis/data_agama_ibata_onedisk_beta3_m200c_hydrabflow/training_data_100000.npz \
        --n-ref 10000 --out OUTDIR
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# --- observed rotation curve (Zhou 2023 u Huang 2016), verbatim from ppc_rotation_curve.py ---
OBS_R_KPC = np.array([
    5.24, 5.74, 6.25, 6.77, 7.23, 7.83, 8.21, 8.78, 9.26, 9.75, 10.25, 10.75, 11.25, 11.75, 12.24,
    12.74, 13.25, 13.74, 14.23, 14.74, 15.23, 15.74, 16.24, 16.74, 17.23, 17.74, 18.35, 18.90,
    19.50, 20.41, 21.28, 22.39, 23.16, 24.00,
])
OBS_VC_KMS = np.array([
    225.10, 233.53, 234.30, 233.17, 236.19, 236.00, 233.19, 233.15, 232.15, 231.24, 230.34, 230.54,
    229.11, 227.48, 226.69, 225.56, 224.90, 223.57, 221.10, 220.19, 219.59, 217.36, 216.61, 217.28,
    216.25, 213.81, 217.53, 212.10, 210.46, 206.69, 207.71, 203.72, 205.20, 200.64,
])
_HUANG = np.array([
    [4.60, 231.24], [5.08, 230.46], [5.58, 230.01], [6.10, 239.61], [6.57, 246.27], [7.07, 243.49],
    [7.58, 242.71], [8.04, 243.23], [8.34, 239.89], [8.65, 237.26], [9.20, 235.30], [9.62, 230.99],
    [10.09, 228.41], [10.58, 224.26], [11.09, 224.94], [11.58, 233.57], [12.07, 240.02],
    [12.73, 242.21], [13.72, 261.78], [14.95, 259.26], [15.52, 268.57], [16.55, 261.17],
    [17.56, 240.66], [18.54, 215.31], [19.50, 214.99], [21.25, 251.68], [23.78, 259.65],
    [26.22, 242.02], [28.71, 224.11], [31.29, 211.20], [33.73, 217.93], [36.19, 219.33],
    [38.73, 213.31], [41.25, 200.05], [43.93, 190.15], [46.43, 198.95], [48.71, 192.91],
    [51.56, 198.90], [57.03, 185.88], [62.55, 173.89], [69.47, 196.36], [79.27, 175.05],
    [98.97, 147.72],
])


def observed_vcirc(split=24.0):
    hr, hv = _HUANG[:, 0], _HUANG[:, 1]
    hi = hr > split
    r = np.concatenate([OBS_R_KPC, hr[hi]])
    vc = np.concatenate([OBS_VC_KMS, hv[hi]])
    return vc[np.argsort(r)]


def _sqd(a, b):
    return (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T


def _mean_sim(q, ref, gamma):
    """Mean RBF kernel similarity of each query row to the reference cloud: (n_q,)."""
    return np.exp(-gamma * _sqd(np.atleast_2d(q), ref)).mean(1)


def median_gamma(ref, cap=2000, rng=None):
    """Median-heuristic bandwidth from a subsample of the reference (pooled pairwise sq-dist)."""
    rng = rng or np.random.default_rng(0)
    sub = ref if len(ref) <= cap else ref[rng.choice(len(ref), cap, replace=False)]
    d = _sqd(sub, sub)
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return 1.0 / (med + 1e-12)


def analyze(name, ref, obs, num_null, rng):
    ref = ref[np.isfinite(ref).all(1)]
    mu = ref.mean(0)
    cov = np.atleast_2d(np.cov(ref, rowvar=False))
    cov += 1e-3 * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
    prec = np.linalg.pinv(cov)
    dz = obs[0] - mu
    d_obs = float(np.sqrt(dz @ prec @ dz))
    d_ref = np.sqrt(np.einsum("ni,ij,nj->n", ref - mu, prec, ref - mu))
    maha_pct = float((d_ref <= d_obs).mean() * 100.0)
    z = (obs[0] - mu) / (ref.std(0) + 1e-9)

    # RBF-MMD with a single-draw bootstrap null. For a single observed potential the reference
    # self-term k(y,y) and the point self-term k(x,x)=1 are identical for the observed point and
    # every null draw, so they cancel in the ranking: MMD is monotone-decreasing in the mean
    # kernel similarity to the reference. p = fraction of single reference draws whose similarity
    # is <= the observed's (i.e. at least as far out). O(n_ref) per draw instead of O(n_ref^2).
    gamma = median_gamma(ref, rng=rng)
    sim_obs = float(_mean_sim(obs, ref, gamma)[0])
    null_idx = rng.integers(0, len(ref), size=num_null)
    sim_null = _mean_sim(ref[null_idx], ref, gamma)
    p = float((sim_null <= sim_obs).mean())
    kyy = float(_mean_sim(ref[rng.choice(len(ref), min(len(ref), 2000), replace=False)],
                          ref, gamma).mean())
    mmd_obs = 1.0 + kyy - 2.0 * sim_obs
    return {
        "n_ref": int(len(ref)), "n_features": int(ref.shape[1]),
        "maha_percentile": round(maha_pct, 2),
        "mean_z": round(float(np.mean(z)), 3),
        "z_min": round(float(z.min()), 3), "z_max": round(float(z.max()), 3),
        "mmd": round(mmd_obs, 5), "p_value": p,
        "obs_bin0": round(float(obs[0, 0]), 2), "ref_bin0_mean": round(float(mu[0]), 2),
        "_d_ref": d_ref, "_d_obs": d_obs, "_sim_null": sim_null, "_sim_obs": sim_obs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="flat training .npz (reference)")
    ap.add_argument("--tv", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "assets", "terminal_velocity.csv"))
    ap.add_argument("--n-ref", type=int, default=10000)
    ap.add_argument("--num-null", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--split", type=float, default=24.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.train, allow_pickle=True)
    ntot = np.asarray(d["vterm_kms"]).shape[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(ntot, min(args.n_ref, ntot), replace=False)

    def ref(k):
        a = np.asarray(d[k], float)
        return a.reshape(a.shape[0], -1)[idx]

    # observed
    import io
    body = "".join(ln for ln in open(args.tv) if not ln.lstrip().startswith("#"))
    tv = np.genfromtxt(io.StringIO(body), delimiter=",", names=True)
    obs = {
        "vterm_kms": tv["vterm_kms"].reshape(1, -1),
        "vcirc_kms": observed_vcirc(args.split).reshape(1, -1),
        "sigma_z": np.array([[71.0]]),
    }

    print(f"Potential-channel misspecification vs {len(idx)} training-set potentials "
          f"(noise-free, per-potential) | num_null={args.num_null}")
    results = {}
    for ch in ("vterm_kms", "vcirc_kms", "sigma_z"):
        R = ref(ch)
        o = obs[ch]
        if R.shape[1] != o.shape[1]:
            print(f"  [skip] {ch}: ref F={R.shape[1]} vs obs F={o.shape[1]}")
            continue
        r = analyze(ch, R, o, args.num_null, rng)
        results[ch] = r
        print(f"  {ch:10s}: n_ref={r['n_ref']:5d} F={r['n_features']:2d} | "
              f"maha_pct={r['maha_percentile']:6.2f} meanz={r['mean_z']:+.2f} "
              f"z[{r['z_min']:+.2f},{r['z_max']:+.2f}] | mmd={r['mmd']:.4f} p={r['p_value']:.3f} | "
              f"obs0={r['obs_bin0']} refmean0={r['ref_bin0_mean']}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chs = list(results)
        colors = {"vterm_kms": "#2c7bb6", "vcirc_kms": "#d7191c", "sigma_z": "#fdae61"}
        fig, axes = plt.subplots(1, len(chs), figsize=(4.6 * len(chs), 3.9), squeeze=False)
        axes = axes[0]
        for ax, ch in zip(axes, chs):
            r = results[ch]
            c = colors.get(ch, "purple")
            ax.hist(r["_d_ref"], bins=60, color="0.75", edgecolor="none",
                    label=f"reference ({r['n_ref']} potentials)")
            ax.axvline(r["_d_obs"], color=c, lw=2.5,
                       label=f"observed MW\n(pct {r['maha_percentile']:.1f})")
            ax.set_title(f"{ch}  (mean z {r['mean_z']:+.2f})", fontsize=10)
            ax.set_xlabel("Mahalanobis distance to reference")
            ax.legend(fontsize=8)
        fig.suptitle(f"Per-channel misspecification vs {len(idx)} training potentials "
                     f"(noise-free, covariance-whitened)", y=1.02)
        fig.tight_layout()
        png = os.path.join(args.out, "mmd_potential_vs_training.png")
        fig.savefig(png, dpi=140, bbox_inches="tight")

        clean = {ch: {k: v for k, v in r.items() if not k.startswith("_")}
                 for ch, r in results.items()}
        with open(os.path.join(args.out, "mmd_potential_vs_training.json"), "w") as f:
            json.dump({"n_ref": len(idx), "channels": clean}, f, indent=2)
        print(f"\nSaved {png}")


if __name__ == "__main__":
    main()
