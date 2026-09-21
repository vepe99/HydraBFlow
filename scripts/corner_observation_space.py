"""Observation-space corner plots: training-set stars (through the training observation model) vs
the real Gaia members, per stream, in the real-fitted great-circle frame.

Six observables per star: phi1, phi2 [deg], parallax [mas], mu_phi1, mu_phi2 [mas/yr], v_los [km/s]
(v_los rows use measured stars only on both sides). Simulated stars from ``--max-rows`` rows per
stream are pooled and drawn as 68/95 % contours + a light scatter; the real members are overplotted
as points. Accepts the flat training layout or a grouped multistream npz.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_fixed_potential_samples import project_sample, stream_frame  # noqa: E402
from ppc_summary_statistics import NAMES, augment_sim  # noqa: E402

LABELS = [("phi1", "phi1 [deg]"), ("phi2", "phi2 [deg]"), ("plx", "parallax [mas]"),
          ("mu_phi1", "mu_phi1 [mas/yr]"), ("mu_phi2", "mu_phi2 [mas/yr]"), ("vlos", "v_los [km/s]")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--simulator", required=True)
    ap.add_argument("--aug", default="stream_global_v5")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=2000, help="rows per stream pushed through the chain")
    ap.add_argument("--max-stars", type=int, default=200000, help="pooled sim stars drawn per stream")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import corner
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    d = np.load(args.real)
    rsim = d["sim_data_projected"]
    rsim = rsim[0] if rsim.ndim == 4 else rsim
    ram = d["attention_mask"]
    ram = ram[:, 0, :] if ram.ndim == 3 else ram
    rvm = d["vlos_mask"]
    rvm = rvm[:, 0, :] if rvm.ndim == 3 else rvm
    jreal = np.asarray(d["j"]).reshape(-1).astype(int)

    ds = np.load(args.sim)
    sd = ds["sim_data_projected"]
    if sd.ndim == 3:
        jf = np.asarray(ds["j"]).reshape(-1).astype(int)
        streams = sorted(set(jf.tolist()))
        n = min(min((jf == s).sum() for s in streams), args.max_rows)
        pick = {s: np.sort(rng.choice(np.flatnonzero(jf == s), n, replace=False)) for s in streams}
        sd = np.stack([sd[pick[s]] for s in streams], axis=1)
        j = np.stack([jf[pick[s]] for s in streams], axis=1)
    else:
        j = np.asarray(ds["j"]).reshape(sd.shape[:2]).astype(int)
        n = min(sd.shape[0], args.max_rows)
        sel = np.sort(rng.choice(sd.shape[0], n, replace=False))
        sd, j = sd[sel], j[sel]
    print(f"{n} rows per stream through '{args.aug}'")
    sd, attn, vm = augment_sim(sd, j, aug_preset=args.aug, simulator=args.simulator, seed=args.seed)

    for col in range(sd.shape[1]):
        jj = int(j[0, col])
        nm = NAMES[jj]
        rrow = int(np.where(jreal == jj)[0][0])
        mem = ram[rrow].astype(bool)
        R, real = stream_frame(rsim[rrow][mem])
        rv = rvm[rrow][mem].astype(bool)
        real_arr = np.column_stack([real[k] for k, _ in LABELS])
        real_arr[~rv, 5] = np.nan

        rows = []
        for i in range(sd.shape[0]):
            m = attn[i, col].astype(bool)
            if m.sum() == 0:
                continue
            s = project_sample(R, sd[i, col][m])
            kv = vm[i, col][m].astype(bool)
            arr = np.column_stack([s[k] for k, _ in LABELS])
            arr[~kv, 5] = np.nan
            rows.append(arr)
        sim_arr = np.concatenate(rows, 0)
        if len(sim_arr) > args.max_stars:
            sim_arr = sim_arr[rng.choice(len(sim_arr), args.max_stars, replace=False)]
        # corner needs finite rows: v_los NaN for unmeasured stars -> plot astrometry from all stars,
        # v_los pairs from measured stars only, by filling NaN with a sentinel range and hiding it
        finite5 = np.isfinite(sim_arr[:, :5]).all(1)
        sim_arr = sim_arr[finite5]
        meas = np.isfinite(sim_arr[:, 5])
        # robust plot ranges from real + sim (0.5-99.5 pct), so a few far outliers do not squash the panels
        ranges = []
        for k in range(6):
            v = np.concatenate([real_arr[:, k][np.isfinite(real_arr[:, k])], sim_arr[:, k][np.isfinite(sim_arr[:, k])]])
            lo, hi = np.percentile(v, [0.5, 99.5])
            pad = 0.1 * (hi - lo)
            ranges.append((lo - pad, hi + pad))
        # v_los: use measured sims only for ALL panels (a consistent star set); astrometry-only
        # panels get the full sample as a second (lighter) layer
        fig = corner.corner(
            sim_arr[meas], labels=[lab for _, lab in LABELS], range=ranges, color="C0",
            plot_datapoints=False, plot_density=False, fill_contours=True,
            levels=(0.68, 0.95), contour_kwargs={"linewidths": 0.8}, hist_kwargs={"density": True},
            smooth=1.0, bins=40,
        )
        axes = np.array(fig.axes).reshape(6, 6)
        rr = real_arr[np.isfinite(real_arr[:, :5]).all(1)]
        for yi in range(6):
            for xi in range(yi):
                ax = axes[yi, xi]
                x, y = rr[:, xi], rr[:, yi]
                ok = np.isfinite(x) & np.isfinite(y)
                ax.scatter(x[ok], y[ok], s=6, color="C3", alpha=0.85, zorder=5)
            ax = axes[yi, yi]
            v = rr[:, yi][np.isfinite(rr[:, yi])]
            ax.hist(v, bins=40, range=ranges[yi], density=True, histtype="step", color="C3", lw=1.2)
        n_meas_real = int(np.isfinite(real_arr[:, 5]).sum())
        fig.suptitle(
            f"{nm}: observation space — blue = {n} training rows through '{args.aug}' "
            f"({meas.sum()} measured-v_los stars of {len(sim_arr)}; 68/95 % contours), red = real Gaia "
            f"({len(rr)} members, {n_meas_real} with v_los)", y=1.01, fontsize=11,
        )
        out = os.path.join(args.out, f"corner_obs_{nm}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
