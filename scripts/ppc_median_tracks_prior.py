"""Prior-predictive check on the BINNED MEDIAN TRACKS: can any prior draw reproduce the real
per-bin medians of (phi2, mu_phi1, mu_phi2, v_los) in the real-fitted stream frame?

Input: a grouped multistream npz (N groups x 3 streams) drawn from the full prior. Each realization
goes through the training observation model (`augment_sim`), is projected into the stream's real
great-circle frame, and its per-phi1-bin medians (equal-count bins of the REAL members) are scored
against the real medians:

    chi2_SEM  = sum_bins,q ((real - sim) / SEM_real)^2        SEM = real per-bin std / sqrt(N_bin)
    d_star    = max_bins,q |real - sim| / std_real             (in units of ONE star's dispersion)

Reports, per stream: the chi2 distribution, how many realizations sit within one star-dispersion
in every bin, the best realization's per-bin offsets, the prior 5-95 % band vs the real track, and
Spearman correlations of chi2 with every varied global/local (which knob moves the track). Also the
per-GROUP joint chi2 (one potential for all three streams). Figures + a JSON report.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import NAMES, augment_sim  # noqa: E402
from plot_fixed_potential_samples import stream_frame, project_sample  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

QS = [("phi2", "deg"), ("mu_phi1", "mas/yr"), ("mu_phi2", "mas/yr"), ("vlos", "km/s")]


def binned(x, y, edges, f=np.nanmedian, minc=3):
    o = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        s = (x >= edges[b]) & ((x < edges[b + 1]) if b < len(edges) - 2 else (x <= edges[b + 1]))
        if s.sum() >= minc:
            o[b] = f(y[s])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--simulator", required=True)
    ap.add_argument("--aug", default="stream_global")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-bins", type=int, default=8)
    ap.add_argument("--k-vlos", type=int, default=3)
    ap.add_argument("--n-best", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-groups", type=int, default=0, help="flat sets: cap the pseudo-groups per stream (0 = all)")
    ap.add_argument("--corner-bins", default="", help="e.g. '0,4,7': also draw a per-stream corner plot of the statistic "
                    "at these phi1 bins for phi2/mu_phi1/mu_phi2 plus every v_los bin (sims = contours, real = red point)")
    ap.add_argument("--stat", choices=["median", "std", "std_mad"], default="median",
                    help="per-bin statistic: median track, plain std (ddof=1) or robust 1.4826*MAD scale")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    K = args.k_bins
    def _std(a):
        return np.std(a, ddof=1)

    def _mad(a):
        return 1.4826 * np.median(np.abs(a - np.median(a)))

    stat = {"median": np.nanmedian, "std": _std, "std_mad": _mad}[args.stat]
    tag = args.stat

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
        # FLAT training set (rows = single streams): regroup per stream into pseudo-groups so the
        # per-stream analysis runs unchanged. Members of a pseudo-group come from DIFFERENT prior
        # draws, so the "joint" score below is meaningless in this mode and is skipped.
        jf = np.asarray(ds["j"]).reshape(-1).astype(int)
        streams = sorted(set(jf.tolist()))
        rows_by = {s_: np.flatnonzero(jf == s_) for s_ in streams}
        rng_g = np.random.default_rng(args.seed)
        n_groups = min(len(v) for v in rows_by.values())
        if args.max_groups:
            n_groups = min(n_groups, args.max_groups)
        pick = {s_: np.sort(rng_g.choice(v, n_groups, replace=False)) for s_, v in rows_by.items()}
        sd = np.stack([sd[pick[s_]] for s_ in streams], axis=1)
        j = np.stack([jf[pick[s_]] for s_ in streams], axis=1)
        n_streams = len(streams)
        flat = True
        print(f"flat set: {len(jf)} rows -> {n_groups} pseudo-groups x {n_streams} streams")

        def param(k, col):
            return np.asarray(ds[k])[pick[streams[col]]].reshape(n_groups, -1)[:, 0]

        cand = [k for k in ds.files if k not in ("sim_data_projected", "j", "vcirc_kms")
                and not k.endswith("_derived") and np.asarray(ds[k]).ndim >= 1
                and np.asarray(ds[k]).shape[0] == len(jf) and np.asarray(ds[k]).size == len(jf)]
        glob_keys = [k for k in cand if np.ptp(np.asarray(ds[k])) > 0]
        loc_keys = []
    else:
        j = np.asarray(ds["j"]).reshape(sd.shape[:2]).astype(int)
        n_groups, n_streams = j.shape
        flat = False
        glob_keys = [k for k in ds.files if ds[k].ndim == 2 and ds[k].shape == (n_groups, 1)
                     and not k.endswith("_derived") and np.ptp(ds[k]) > 0]
        loc_keys = [k for k in ("t_end", "m_progenitor", "vr", "r", "mu_ra_cosdec", "mu_dec")
                    if k in ds.files and np.ptp(ds[k], axis=0).max() > 0]

        def param(k, col):
            a = np.asarray(ds[k])
            return a[:, 0] if a.ndim == 2 else a[:, col, 0]

    sd, attn, vm = augment_sim(sd, j, aug_preset=args.aug, simulator=args.simulator, seed=args.seed)

    report = {"n_groups": int(n_groups), "streams": {}}
    chi2_group = np.zeros(n_groups)
    fig, axs = plt.subplots(len(QS), n_streams, figsize=(5 * n_streams, 3.2 * len(QS)))
    for col in range(n_streams):
        jj = int(j[0, col])
        nm = NAMES[jj]
        rrow = int(np.where(jreal == jj)[0][0])
        mem = ram[rrow].astype(bool)
        R, real = stream_frame(rsim[rrow][mem])
        rv = rvm[rrow][mem].astype(bool)
        # phi1 bins: equal-count quantiles of the REAL members; v_los gets its own coarser grid from the
        # MEASURED real stars (as the summary statistics do), since both real and sim have few of them.
        edges_all = np.quantile(real["phi1"], np.linspace(0, 1, K + 1))
        edges_v = np.quantile(real["phi1"][rv], np.linspace(0, 1, args.k_vlos + 1))
        EDG = {q: (edges_v if q == "vlos" else edges_all) for q, _ in QS}
        MID = {q: 0.5 * (e[1:] + e[:-1]) for q, e in EDG.items()}
        RT, RS, RN = {}, {}, {}
        for q, _ in QS:
            m = rv if q == "vlos" else np.ones(len(rv), bool)
            e = EDG[q]
            RT[q] = binned(real["phi1"][m], real[q][m], e, stat)
            RS[q] = binned(real["phi1"][m], real[q][m], e, lambda a: np.std(a, ddof=1))
            RN[q] = binned(real["phi1"][m], real["phi1"][m], e, len, minc=1)
            # standard error of the real statistic: std/sqrt(N) for a median-like location, std/sqrt(2(N-1)) for a scale
            SE = {qq: (RS[qq] / np.sqrt(RN[qq]) if args.stat == "median" else RS[qq] / np.sqrt(2 * (RN[qq] - 1))) for qq in RS}
        T = {q: np.full((n_groups, len(EDG[q]) - 1), np.nan) for q, _ in QS}
        n_att = np.zeros(n_groups, int)
        for i in range(n_groups):
            sel = attn[i, col].astype(bool)
            n_att[i] = sel.sum()
            if n_att[i] < 3 * K:
                continue
            s = project_sample(R, sd[i, col][sel])
            kv = vm[i, col][sel].astype(bool)
            for q, _ in QS:
                m = kv if q == "vlos" else slice(None)
                T[q][i] = binned(s["phi1"][m], s[q][m], EDG[q], stat)
        # (SE is defined inside the loop per stream)
        # scores over bins where BOTH the real and the sim statistic are defined; chi2 is rescaled to
        # the full number of real bins so a sparsely populated realization is not rewarded, and the
        # fraction of missing sim bins is reported.
        chi2 = np.zeros(n_groups)
        dstar = np.zeros(n_groups)
        chi2_q = {}
        miss = {}
        for q, _ in QS:
            ok = np.isfinite(RT[q])
            sem = SE[q][ok]
            r = (RT[q][ok][None, :] - T[q][:, ok]) / sem[None, :]
            fin = np.isfinite(r)
            n_used = fin.sum(1)
            miss[q] = 1.0 - n_used / max(ok.sum(), 1)
            c = np.where(fin, r, 0.0) ** 2
            chi2_q[q] = np.where(n_used > 0, c.sum(1) * ok.sum() / np.maximum(n_used, 1), np.inf)
            chi2 += chi2_q[q]
            den = RS[q][ok][None, :] if args.stat == "median" else RT[q][ok][None, :]
            rs = np.abs(RT[q][ok][None, :] - T[q][:, ok]) / den
            dstar = np.maximum(dstar, np.where(fin.any(1), np.nanmax(np.where(fin, rs, np.nan), axis=1), np.inf))
        valid = n_att >= 3 * K
        chi2[~valid] = np.inf
        dstar[~valid] = np.inf
        chi2_group += np.where(valid, chi2, 1e6)
        order = np.argsort(chi2)
        best = order[: args.n_best]
        n_dof = int(sum(np.isfinite(RT[q]).sum() for q, _ in QS))
        rep = {"stat": tag, "n_valid": int(valid.sum()), "n_dof": n_dof, "median_frac_missing_sim_bins": {q: float(np.median(miss[q][valid])) for q, _ in QS},
               "chi2_SEM": {"min": float(chi2[order[0]]), "p5": float(np.percentile(chi2[valid], 5)), "median": float(np.median(chi2[valid]))},
               "frac_within_1_star_sigma_all_bins": float((dstar < 1).mean()), "frac_within_2_star_sigma_all_bins": float((dstar < 2).mean()),
               "best_group": int(order[0]), "best_dstar": float(dstar[order[0]]),
               "best_per_bin_offset": {q: np.round(RT[q] - T[q][order[0]], 3).tolist() for q, _ in QS},
               "best_params": {k: float(param(k, col)[order[0]]) for k in glob_keys + loc_keys},
               "spearman_chi2_vs_param": {}}
        for k in glob_keys + loc_keys:
            rho = spearmanr(param(k, col)[valid], np.log(chi2[valid])).statistic
            rep["spearman_chi2_vs_param"][k] = round(float(rho), 3)
        for q, _ in QS:
            rq = np.abs(RT[q][None] - T[q]) / (RS[q][None] if args.stat == "median" else RT[q][None])
            okq = np.isfinite(rq).any(1)
            if args.stat != "median":
                # cold/hot balance per bin: fraction of draws whose dispersion is BELOW the real one
                rep_extra = rep.setdefault("frac_sim_below_real_per_bin", {})
                rep_extra[q] = np.round(np.nanmean(T[q][valid] < RT[q][None], axis=0), 2).tolist()
            dq = np.where(okq, np.nanmax(np.where(np.isfinite(rq), rq, np.nan), axis=1), np.inf)
            rep[f"chi2_{q}_min"] = float(chi2_q[q][order[0]])
            rep[f"frac_{q}_within_1_star_sigma"] = float((dq < 1)[valid].mean())
        report["streams"][nm] = rep
        if args.corner_bins:
            import corner as _corner

            cb = [int(b) for b in args.corner_bins.split(",")]
            cols, names, real_vals, real_err = [], [], [], []
            for q, unit in QS:
                bins = cb if q != "vlos" else list(range(T[q].shape[1]))
                for b in bins:
                    if b >= T[q].shape[1] or not np.isfinite(RT[q][b]):
                        continue
                    cols.append(T[q][:, b])
                    names.append(f"{q}\nbin {b} [{unit}]")
                    real_vals.append(RT[q][b])
                    real_err.append(SE[q][b])
            X = np.column_stack(cols)[valid]
            # keep a statistic only if most rows have it (M68's v_los bins are empty in ~2/3 of
            # the rows: requiring them would keep the few rows with many measured stars -> biased)
            frac_ok = np.isfinite(X).mean(0)
            keep_c = frac_ok >= 0.5
            dropped = [names[k].replace("\n", " ") for k in range(len(names)) if not keep_c[k]]
            X = X[:, keep_c]
            names = [n_ for n_, k_ in zip(names, keep_c) if k_]
            real_vals = [v_ for v_, k_ in zip(real_vals, keep_c) if k_]
            real_err = [e_ for e_, k_ in zip(real_err, keep_c) if k_]
            okrow = np.isfinite(X).all(1)
            X = X[okrow]
            rng_c = [(min(np.percentile(X[:, k], 0.5), real_vals[k]) - 0.05 * np.ptp(X[:, k]),
                      max(np.percentile(X[:, k], 99.5), real_vals[k]) + 0.05 * np.ptp(X[:, k])) for k in range(X.shape[1])]
            figc = _corner.corner(X, labels=names, range=rng_c, color="C0", plot_datapoints=False,
                                  plot_density=False, fill_contours=True, levels=(0.68, 0.95),
                                  smooth=1.0, bins=35, truths=real_vals, truth_color="C3",
                                  label_kwargs={"fontsize": 8})
            axc = np.array(figc.axes).reshape(len(names), len(names))
            for k in range(len(names)):
                axc[k, k].axvspan(real_vals[k] - real_err[k], real_vals[k] + real_err[k], color="C3", alpha=0.25)
                for kk in range(k):
                    axc[k, kk].errorbar(real_vals[kk], real_vals[k], xerr=real_err[kk], yerr=real_err[k],
                                        fmt="o", color="C3", ms=4, capsize=2, zorder=6)
            sub = f"; dropped (finite in <50 % of rows): {', '.join(dropped)}" if dropped else ""
            figc.suptitle(f"{nm}: per-bin {tag} as observables — {len(X)} prior rows (68/95 %), "
                          f"red = real Gaia ± SE{sub}", y=1.01, fontsize=12)
            figc.savefig(os.path.join(args.out, f"corner_{tag}_{nm}.png"), dpi=100, bbox_inches="tight")
            plt.close(figc)
        for k, (q, unit) in enumerate(QS):
            ax = axs[k, col]
            mid = MID[q]
            lo, hi = np.nanpercentile(T[q][valid], [5, 95], 0)
            l1, h1 = np.nanpercentile(T[q][valid], [16, 84], 0)
            ax.fill_between(mid, lo, hi, color="C0", alpha=0.15, label="prior 5-95 %")
            ax.fill_between(mid, l1, h1, color="C0", alpha=0.3, label="prior 16-84 %")
            for b in best:
                ax.plot(mid, T[q][b], color="C1", alpha=0.6, lw=0.9)
            ax.plot([], [], color="C1", label=f"best {args.n_best} (chi2)")
            ax.errorbar(mid, RT[q], yerr=SE[q], fmt="ko", ms=4, capsize=2, label=f"real {tag} ± SE")
            ax.set_ylabel(f"{q} [{unit}]")
            if k == 0:
                ax.set_title(f"{nm}  (n={int(valid.sum())}, best chi2={chi2[order[0]]:.0f}/{n_dof} dof)")
            if k == len(QS) - 1:
                ax.set_xlabel("phi1 [deg]")
            if k == 0 and col == 0:
                ax.legend(fontsize=7)
    gvalid = chi2_group < 1e5
    go = np.argsort(chi2_group)
    if flat:
        report["joint"] = None  # pseudo-groups mix prior draws: no joint score in flat mode
    else:
        report["joint"] = {"best_group": int(go[0]), "best_chi2_sum": float(chi2_group[go[0]]),
                           "median_chi2_sum": float(np.median(chi2_group[gvalid])),
                           "best_params": {k: float(param(k, 0)[go[0]]) for k in glob_keys}}
    fig.suptitle(f"Prior-predictive binned {tag} tracks — {os.path.basename(args.sim)} ({args.simulator})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{tag}_tracks_prior.png"), dpi=130)
    json.dump(report, open(os.path.join(args.out, f"{tag}_tracks_prior.json"), "w"), indent=1)
    for nm, r in report["streams"].items():
        print(f"\n== {nm}: valid {r['n_valid']}, dof {r['n_dof']}, chi2 min/p5/median = {r['chi2_SEM']['min']:.0f}/{r['chi2_SEM']['p5']:.0f}/{r['chi2_SEM']['median']:.0f}; "
              f"within 1 (2) [star-sigma | x real] in ALL bins: {100*r['frac_within_1_star_sigma_all_bins']:.1f}% ({100*r['frac_within_2_star_sigma_all_bins']:.1f}%)")
        print("   per-quantity frac within 1 star-sigma:", {q: round(r[f'frac_{q}_within_1_star_sigma'], 3) for q, _ in QS}, "| median frac of missing sim bins:", {q: round(v, 2) for q, v in r['median_frac_missing_sim_bins'].items()})
        print("   best group", r["best_group"], "d* =", round(r["best_dstar"], 2), "per-bin offsets:", {q: r["best_per_bin_offset"][q] for q, _ in QS})
        if "frac_sim_below_real_per_bin" in r:
            print("   P(sim < real) per bin:", r["frac_sim_below_real_per_bin"])
        print("   best params:", {k: (round(v, 3) if abs(v) < 1e4 else f"{v:.3g}") for k, v in r["best_params"].items()})
        print("   spearman(log chi2, param):", dict(sorted(r["spearman_chi2_vs_param"].items(), key=lambda kv: -abs(kv[1]))))
    if report["joint"] is not None:
        print("\n== joint (one potential, 3 streams): best group", report["joint"]["best_group"], "chi2 sum", round(report["joint"]["best_chi2_sum"]), "median", round(report["joint"]["median_chi2_sum"]))
        print("   best params:", {k: (round(v, 3) if abs(v) < 1e4 else f"{v:.3g}") for k, v in report["joint"]["best_params"].items()})


if __name__ == "__main__":
    main()
