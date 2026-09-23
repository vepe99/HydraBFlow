#!/usr/bin/env python
"""Nearest training rows to the real members in the BINNED summary-statistic space (the 14-channel
`sim_summary` grid the network ingests), and the k nearest drawn in bin values.

Per stream: training rows through the training chain up to `stream_summary_grid`, the real members
through the real preset's grid step (same frame / edges / estimator). Distance = mean over cells
valid on BOTH sides (occupancy >= summary_min_count) of ((sim - real) / MAD_cell)^2 on the 10
STATISTIC channels (occupancy is OOD by construction — real bins are equal-count — so it is not
ranked on unless --with-occupancy). Rows with < --min-valid-frac of the real's valid cells are
excluded. Outputs: nearest_bins_<stream>.png (10 channels x K bins: prior band, k nearest, real),
nearest_bins_params_<stream>.png, nearest_training_bins.json, nn_grids_<stream>.npz.

  .venv/bin/python scripts/nearest_training_bins.py --out outputs/nn_v4_rnbody_palau23_bins
"""
import argparse, json, os, sys
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0"); os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_grid_coverage import NAMES, STAT, OCC, compose_aug, run_chain  # noqa: E402
from ppc_observation_space import read_rows  # noqa: E402
from nearest_training_streams import GLOBALS, LOCALS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/data_jarvis/data_agama_rnbody_ibata_m200c_v4_hydrabflow/training_data_100000.npz")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_palau23_dr3.npz")
    ap.add_argument("--simulator", default="stream_agama_rnbody_ibata_m200c_v4")
    ap.add_argument("--aug", default="stream_global_palau23_dr3_emperr")
    ap.add_argument("--real-aug", default="stream_real_global_palau23_dr3_emperr")
    ap.add_argument("--override", action="append", default=[], help="Hydra override(s), e.g. augmentation.params.summary_scale=std")
    ap.add_argument("--n-rows", type=int, default=100000, help="training rows per stream (clamped to the set)")
    ap.add_argument("--k", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-occupancy", action="store_true"); ap.add_argument("--min-valid-frac", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=5000, help="rows per augmentation call (memory)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)

    aug = compose_aug(a.simulator, a.aug, a.override)
    steps = [str(s) for s in aug.steps]; steps = steps[: steps.index("stream_summary_grid") + 1]
    steps = [s for s in steps if s not in ("add_noise_to_vcirc", "log10_vcirc")]
    min_count = int(aug.params.get("summary_min_count", 3))

    with np.load(a.train) as d:
        jf = np.asarray(d["j"]).reshape(-1).astype(int)
        params = {k: np.asarray(d[k]).reshape(len(jf), -1)[:, 0] for k in GLOBALS + LOCALS if k in d.files}
    n_rows = min(a.n_rows, *[int((jf == j).sum()) for j in range(3)])
    rows = np.concatenate([np.sort(rng.choice(np.flatnonzero(jf == j), size=n_rows, replace=False)) for j in range(3)])
    grids = []
    for c0 in range(0, len(rows), a.chunk):
        r = rows[c0:c0 + a.chunk]
        batch = {"sim_data_projected": read_rows(a.train, "sim_data_projected", r).astype(np.float32), "j": jf[r].reshape(-1, 1).astype(np.float32)}
        grids.append(np.asarray(run_chain(aug, steps, batch, a.seed + c0)["sim_summary"]))
        print(f"[grid] {c0 + len(r)}/{len(rows)} rows", flush=True)
    ssum = np.concatenate(grids); sj = jf[rows]                                  # (N, K, 14)

    raug = compose_aug(a.simulator, a.real_aug, a.override); d = np.load(a.real)
    m = int(np.asarray(d["j"]).size); rb = {}
    for k in d.files:
        if k == "j": continue
        x = np.asarray(d[k], dtype=np.float32)
        if x.ndim >= 3 and x.shape[0] == 1 and x.shape[1] == m: x = x.reshape(m, *x.shape[2:])
        if k in ("attention_mask", "vlos_mask") and x.ndim == 2: x = x[:, None, :]
        rb[k] = x
    rb["j"] = np.asarray(d["j"]).reshape(m, 1).astype(np.float32)
    rsum = np.asarray(run_chain(raug, ["stream_summary_grid"], rb, a.seed)["sim_summary"]); rj = rb["j"].reshape(-1).astype(int)
    K = ssum.shape[1]; labels = STAT + (OCC if a.with_occupancy else [])
    chans = list(range(len(STAT))) + ([10, 11] if a.with_occupancy else [])

    def valid(A, ntr, nvl):
        """(…,K,10) validity of the statistic cells from the occupancy channels."""
        V = np.zeros(A.shape[:-1] + (len(STAT),), bool)
        for ci, lab in enumerate(STAT):
            V[..., ci] = (nvl if lab.endswith("vlos") else ntr) >= min_count
        return V

    rep = dict(train=a.train, real=a.real, aug=a.aug, n_rows=int(n_rows), k=a.k, with_occupancy=a.with_occupancy, override=a.override)
    for row in range(m):
        j = int(rj[row]); name = NAMES[j]; idx = rows[sj == j]
        S = ssum[sj == j].astype(float); R = rsum[row].astype(float)                     # (n,K,14), (K,14)
        Sv = valid(S, S[..., 10], S[..., 11]); Rv = valid(R, R[:, 10], R[:, 11])           # (n,K,10), (K,10)
        Sst = np.where(Sv, S[..., :10], np.nan); Rst = np.where(Rv, R[:, :10], np.nan)
        mad = 1.4826 * np.nanmedian(np.abs(Sst - np.nanmedian(Sst, 0)), 0) + 1e-9         # (K,10)
        Z2 = ((Sst - Rst) / mad) ** 2                                                       # (n,K,10)
        if a.with_occupancy:
            So, Ro = S[..., 10:12], R[:, 10:12]; mo = np.maximum(1.4826 * np.median(np.abs(So - np.median(So, 0)), 0), 1.0)  # counts: floor at 1 star
            Z2 = np.concatenate([Z2, ((So - Ro) / mo) ** 2], -1)
        both = np.isfinite(Z2); n_real = np.isfinite(Rst).sum() + (2 * K if a.with_occupancy else 0)
        frac = both.sum((1, 2)) / max(n_real, 1)
        dist = np.where(frac >= a.min_valid_frac, np.nanmean(np.where(both, Z2, np.nan), (1, 2)), np.inf)
        order = np.argsort(dist); nn = order[: a.k]; dnn = dist[nn]
        # null: sim-vs-sim distance with the same metric on a 60-row pool
        pick = rng.choice(np.flatnonzero(np.isfinite(dist)), size=min(60, int(np.isfinite(dist).sum())), replace=False)
        null = []
        for p in pick:
            Zp = ((Sst[pick] - Sst[p]) / mad) ** 2; b = np.isfinite(Zp); ok = b.sum((1, 2)) >= a.min_valid_frac * np.isfinite(Sst[p]).sum()
            dd = np.where(ok, np.nanmean(np.where(b, Zp, np.nan), (1, 2)), np.inf); dd[pick == p] = np.inf; null.append(dd.min())
        null = np.array(null); real_pool = float(dist[pick].min())
        out = dict(n_usable=int(np.isfinite(dist).sum()), n_real_cells=int(n_real), nearest_chi2=float(dnn[0]), kth_chi2=float(dnn[-1]),
                   median_chi2=float(np.median(dist[np.isfinite(dist)])), real_nearest_in_pool=real_pool,
                   sim_nearest_in_pool_median=float(np.median(null)), real_nearest_pool_pct=float(100 * np.mean(null <= real_pool)),
                   nn_rows=idx[nn].tolist(), nn_chi2=dnn.tolist(), params={})
        print(f"\n== {name}: {out['n_usable']} usable rows, {n_real} real cells; nearest mean chi2/cell {dnn[0]:.3f}, k-th {dnn[-1]:.3f}, median {out['median_chi2']:.2f}; "
              f"60-row pool: real->nearest {real_pool:.3f} vs sim->nearest median {np.median(null):.3f} (real at the {out['real_nearest_pool_pct']:.0f}th pct)")
        print(f"  {'param':36s} {'prior med [16,84]':>28s}   {'k-NN med [16,84]':>28s}  shift/prior_sd")
        for k in GLOBALS + LOCALS:
            if k not in params: continue
            pv, nv = params[k][jf == j], params[k][idx[nn]]; pv, nv = pv[np.isfinite(pv)], nv[np.isfinite(nv)]
            if not len(nv) or pv.std() == 0: continue
            pq, nq = np.percentile(pv, [50, 16, 84]), np.percentile(nv, [50, 16, 84]); z = (nq[0] - pq[0]) / pv.std()
            out["params"][k] = dict(prior=pq.tolist(), nn=nq.tolist(), shift_sd=float(z))
            print(f"  {k:36s} {pq[0]:9.3g} [{pq[1]:8.3g},{pq[2]:8.3g}]   {nq[0]:9.3g} [{nq[1]:8.3g},{nq[2]:8.3g}]  {z:+.2f}")
        rep[name] = out
        np.savez(os.path.join(a.out, f"nn_grids_{name}.npz"), rows=idx[nn], chi2=dnn, sim_summary=S[nn], real_summary=R, prior_p16_50_84=np.nanpercentile(Sst, [16, 50, 84], 0))

        # --- bin-value figure: 12 channels (10 stats + 2 occupancy) x K bins
        fig, ax = plt.subplots(3, 4, figsize=(20, 11)); ax = ax.ravel(); b = np.arange(K)
        for ci, lab in enumerate(STAT + OCC):
            A = Sst[..., ci] if ci < 10 else S[..., ci]
            lo, md, hi = np.nanpercentile(A, [2.5, 50, 97.5], 0)
            ax[ci].fill_between(b, lo, hi, color="0.85", label="prior 95 % (valid cells)")
            ax[ci].plot(b, md, color="0.5", lw=1, label="prior median")
            An = A[nn]
            for r_ in range(len(nn)):
                ax[ci].plot(b, An[r_], color="C0", alpha=min(1.0, 8.0 / len(nn)), lw=0.8)
            ax[ci].plot(b, np.nanmedian(An, 0), color="C0", lw=2, label=f"{len(nn)} nearest (median)")
            rv = Rst[:, ci] if ci < 10 else R[:, ci]
            ax[ci].plot(b, rv, "kx-", lw=1.5, ms=8, label="real")
            ax[ci].set_title(lab + ("" if ci < 10 else "  [not ranked on]" if not a.with_occupancy else ""), fontsize=10); ax[ci].set_xlabel("phi1 bin")
        ax[0].legend(fontsize=7); fig.suptitle(f"{name}: {len(nn)} nearest training rows in the binned summary space (mean chi2/cell {dnn[0]:.2f}-{dnn[-1]:.2f}); {a.aug}")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"nearest_bins_{name}.png"), dpi=120); plt.close(fig)

        keys = [k for k in out["params"] if k in GLOBALS]
        fig, ax = plt.subplots(2, 5, figsize=(20, 7)); ax = ax.ravel()
        for i, k in enumerate(keys[:10]):
            pv, nv = params[k][jf == j], params[k][idx[nn]]
            ax[i].hist(pv[np.isfinite(pv)], bins=30, density=True, color="0.7", label="prior"); ax[i].hist(nv[np.isfinite(nv)], bins=15, density=True, histtype="step", color="C3", lw=2, label=f"{a.k} nearest")
            ax[i].set_title(k, fontsize=9)
        ax[0].legend(fontsize=7); fig.suptitle(f"{name}: globals of the {a.k} nearest rows (binned space) vs prior")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"nearest_bins_params_{name}.png"), dpi=110); plt.close(fig)
    json.dump(rep, open(os.path.join(a.out, "nearest_training_bins.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
