#!/usr/bin/env python
"""Coverage check of the BINNED network input (``sim_summary`` from ``stream_summary_grid``).

Complements ``ppc_summary_coverage.py`` (which bins with the PPC scripts' own estimator): here both
sides go through the augmentation chains the network actually sees — the TRAINING preset up to
``stream_summary_grid`` for the simulated groups, the REAL preset's ``stream_summary_grid`` (same
frame, edges, MAD scale, min_count, occupancy channels) for the Gaia members — and every real
(stream, channel, phi1-bin) cell is ranked inside the simulated distribution of that cell. Cells
whose real occupancy is below ``summary_min_count`` (value substituted 0) are skipped; simulated
cells below it are treated as missing, exactly as the masked backbone does.

Outputs (--out): summary_grid_coverage_percentiles.png (heatmap incl. the two occupancy channels),
summary_grid_coverage.json (per-cell real / sim quantiles / percentile / robust z).

Usage:
  uv run python scripts/ppc_summary_grid_coverage.py --sim <grouped npz (N,S,P,6)> --out <dir> \
      --simulator stream_agama_rnbody_ibata_m200c_v4 [--aug stream_global_ibata_grid_v2] \
      [--real-aug stream_real_global_ibata_grid_v2]
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

NAMES = ["Pal5", "NGC3201", "M68"]
STAT = ["med_phi2", "std_phi2", "med_plx", "std_plx", "med_mu_phi1", "std_mu_phi1",
        "med_mu_phi2", "std_mu_phi2", "med_vlos", "std_vlos"]
OCC = ["n_track", "n_vlos"]


def compose_aug(simulator, aug_preset, extra=()):
    from hydra import compose, initialize_config_dir
    from hydrabflow.config import register_configs

    register_configs()
    conf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config",
                      overrides=[f"simulator={simulator}", f"augmentation={aug_preset}", "composition=global", *extra])
    aug = OmegaConf.create(OmegaConf.to_container(cfg.augmentation, resolve=True))
    repo = os.path.dirname(conf_dir)
    res = str(aug.params.get("resources_dir", "data"))
    if not os.path.exists(os.path.join(res, str(aug.params.get("member_table", "apjad382dt1_mrt.txt")))):
        aug.params["resources_dir"] = os.path.join(repo, "assets", "gaia")
    return aug


def run_chain(aug, steps, batch, seed):
    from hydrabflow.registry import build_augmentations

    aug = OmegaConf.create(OmegaConf.to_container(aug, resolve=True))
    aug.steps = list(steps)
    for fn in build_augmentations(aug, np.random.default_rng(seed), context={}):
        batch = fn(batch)
    return batch


def rank(sim, real):
    s = sim[np.isfinite(sim)]
    if s.size < 10 or not np.isfinite(real):
        return np.nan, np.nan
    mad = 1.4826 * np.median(np.abs(s - np.median(s)))
    return 100.0 * np.mean(s <= real), (real - np.median(s)) / mad if mad > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--simulator", default="stream_agama_rnbody_ibata_m200c_v4")
    ap.add_argument("--aug", default="stream_global_ibata_grid_v2")
    ap.add_argument("--real-aug", default="stream_real_global_ibata_grid_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="")
    ap.add_argument("--override", action="append", default=[], help="extra Hydra override(s), e.g. augmentation.params.summary_scale=std")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- simulated side: training chain up to the grid summary
    aug = compose_aug(args.simulator, args.aug, args.override)
    steps = [str(s) for s in aug.steps]
    steps = steps[: steps.index("stream_summary_grid") + 1]
    steps = [s for s in steps if s not in ("add_noise_to_vcirc", "log10_vcirc")]  # vcirc not carried here
    sd = np.load(args.sim)
    x = sd["sim_data_projected"]; n, s_, p, c = x.shape
    jj = np.asarray(sd["j"]).reshape(n * s_, 1).astype(np.float32)
    batch = {"sim_data_projected": np.asarray(x, np.float32).reshape(n * s_, p, c), "j": jj}
    batch = run_chain(aug, steps, batch, args.seed)
    ssum = np.asarray(batch["sim_summary"])  # (n*s, K, 14)
    min_count = int(aug.params.get("summary_min_count", 3))
    K, C = ssum.shape[1], ssum.shape[2]
    assert C == 14, f"expected the 14-channel grid layout, got {C}"

    # --- real side: the real preset's grid step on the Gaia members (same frame/edges/estimator)
    raug = compose_aug(args.simulator, args.real_aug, args.override)
    d = np.load(args.real)
    m = int(np.asarray(d["j"]).size); max_p = int(aug.params.get("max_particles", 300))
    rb = {}
    for k in d.files:
        a = np.asarray(d[k], dtype=np.float32) if k != "j" else None
        if k == "j":
            continue
        a = a[:, :max_p] if a.ndim == 2 else a[:, :, :max_p]
        if a.ndim >= 3 and a.shape[0] == 1 and a.shape[1] == m:
            a = a.reshape(m, *a.shape[2:])
        if k in ("attention_mask", "vlos_mask") and a.ndim == 2:
            a = a[:, None, :]
        rb[k] = a
    rb["j"] = np.asarray(d["j"]).reshape(m, 1).astype(np.float32)
    rb = run_chain(raug, ["stream_summary_grid"], rb, args.seed)
    rsum = np.asarray(rb["sim_summary"])  # (m, K, 14)
    rj = rb["j"].reshape(-1).astype(int); sj = jj.reshape(-1).astype(int)

    labels = STAT + OCC
    report = {"sim": args.sim, "n_groups": int(n), "min_count": min_count, "streams": {}}
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    for row in range(m):
        j = int(rj[row]); name = NAMES[j]
        S = ssum[sj == j]                     # (n_j, K, 14)
        R = rsum[row]                         # (K, 14)
        s_ntr, s_nvl = S[:, :, 10], S[:, :, 11]
        r_ntr, r_nvl = R[:, 10], R[:, 11]
        pct = np.full((len(labels), K), np.nan); zz = np.full_like(pct, np.nan); cells = {}
        for ci, lab in enumerate(labels):
            ch = ci
            is_vlos = lab.endswith("vlos") and not lab.startswith("n_")
            s_occ = s_nvl if is_vlos else s_ntr; r_occ = r_nvl if is_vlos else r_ntr
            vals = S[:, :, ch].astype(float).copy()
            if lab in STAT:
                vals[s_occ < min_count] = np.nan      # substituted zeros are not measurements
            per = []
            for b in range(K):
                rv = float(R[b, ch])
                if lab in STAT and r_occ[b] < min_count:
                    per.append(dict(real=rv, skipped="real occupancy < min_count")); continue
                pc, z = rank(vals[:, b], rv); pct[ci, b], zz[ci, b] = pc, z
                fin = vals[:, b][np.isfinite(vals[:, b])]
                per.append(dict(real=rv, percentile=pc, robust_z=z, n_sim=int(fin.size),
                                sim_p2_5=float(np.percentile(fin, 2.5)) if fin.size else None,
                                sim_p50=float(np.percentile(fin, 50)) if fin.size else None,
                                sim_p97_5=float(np.percentile(fin, 97.5)) if fin.size else None))
            cells[lab] = per
        fin = np.isfinite(pct)
        in95 = float(((pct > 2.5) & (pct < 97.5))[fin].mean()); in99 = float(((pct > 0.5) & (pct < 99.5))[fin].mean())
        out95 = [(labels[i], int(b), round(float(pct[i, b]), 1)) for i, b in zip(*np.where(fin & ~((pct > 2.5) & (pct < 97.5))))]
        # --- corner of headline scalars of the binned input (sims grey, real red)
        def scalars(A, ntr, nvl):  # A: (..., K, 14) -> (..., n_scalars)
            A = A.astype(float).copy()
            for ci, lab in enumerate(STAT):
                occ = nvl if lab.endswith("vlos") else ntr
                A[..., ci][occ < min_count] = np.nan
            mid = slice(2, K - 2)
            with np.errstate(all="ignore"):
                cols = [ntr[..., mid].mean(-1), nvl[..., mid].mean(-1), ntr.sum(-1),
                        np.sqrt(np.nanmean(A[..., 0] ** 2, -1)),
                        np.nanmedian(A[..., 1], -1), np.nanmedian(A[..., 3], -1), np.nanmedian(A[..., 5], -1),
                        np.nanmedian(A[..., 7], -1), np.nanmedian(A[..., 9], -1)]
            return np.stack(cols, -1)
        clabels = [f"n_track\ncentral bins mean", f"n_vlos\ncentral bins mean", "n_track\nsum (in-range stars)",
                   "rms med phi2\n[deg]", "median std phi2\n[deg]", "median std plx\n[mas]",
                   "median std mu_phi1\n[mas/yr]", "median std mu_phi2\n[mas/yr]", "median std vlos\n[km/s]"]
        sc_sim = scalars(S, s_ntr, s_nvl); sc_real = scalars(R[None], r_ntr[None], r_nvl[None])[0]
        keep = np.isfinite(sc_sim).all(1) & np.isfinite(sc_real)[None, :].all(1)
        cols_ok = np.isfinite(sc_real) & (np.isfinite(sc_sim).mean(0) > 0.5)
        X = sc_sim[:, cols_ok]; good = np.isfinite(X).all(1); X = X[good]
        try:
            import corner as _corner
            lo, hi = np.nanpercentile(X, 0.5, 0), np.nanpercentile(X, 99.5, 0)
            rv = sc_real[cols_ok]; lo = np.minimum(lo, rv); hi = np.maximum(hi, rv)
            rng_ = [(l - 0.05 * (h - l), h + 0.05 * (h - l)) for l, h in zip(lo, hi)]
            cf = _corner.corner(X, labels=[c for c, k in zip(clabels, cols_ok) if k], truths=rv, truth_color="red",
                                color="0.3", range=rng_, quantiles=[0.025, 0.975], bins=40,
                                plot_datapoints=True, data_kwargs=dict(alpha=0.15, ms=1.5),
                                label_kwargs=dict(fontsize=8), title_kwargs=dict(fontsize=8))
            cf.suptitle(f"{name}: binned network-input scalars from {X.shape[0]} sims (grey, dashed = central 95 %) "
                        f"vs the real Gaia value (red). {args.title}", fontsize=9)
            cf.savefig(os.path.join(args.out, f"summary_grid_corner_{name}.png"), dpi=110, bbox_inches="tight")
            plt.close(cf)
        except ImportError:
            print("corner not installed; skipping corner plot")
        report["streams"][name] = dict(n_sim=int(S.shape[0]), inside95=in95, inside99=in99, outside95=out95,
                                       n_cells=int(fin.sum()), cells=cells,
                                       real_occupancy=dict(n_track=r_ntr.tolist(), n_vlos=r_nvl.tolist()),
                                       sim_occupancy_median=dict(n_track=np.median(s_ntr, 0).tolist(),
                                                                 n_vlos=np.median(s_nvl, 0).tolist()))
        print(f"{name:8s} sims {S.shape[0]:5d}  cells {int(fin.sum())}  inside 95%: {100*in95:.0f}%  "
              f"(99%: {100*in99:.0f}%)  outside95: {out95}")
        ax = axes[row]
        im = ax.imshow(pct, aspect="auto", cmap="coolwarm", vmin=0, vmax=100)
        for i in range(len(labels)):
            for b in range(K):
                if np.isfinite(pct[i, b]):
                    ax.text(b, i, f"{pct[i, b]:.0f}", ha="center", va="center", fontsize=7,
                            color="k" if 10 < pct[i, b] < 90 else "w")
                elif i < len(STAT):
                    ax.text(b, i, "·", ha="center", va="center", fontsize=9, color="grey")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(range(K)); ax.set_xlabel("phi1 bin")
        ax.set_title(f"{name}: percentile of the real sim_summary cell inside {S.shape[0]} sims  "
                     f"(inside 95%: {100*in95:.0f}%, 99%: {100*in99:.0f}%; '·' = real occupancy < {min_count})", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02, label="percentile of real value in sim distribution")
    fig.suptitle(args.title or f"Binned network-input coverage ({args.aug} / {args.real_aug})")
    fig.savefig(os.path.join(args.out, "summary_grid_coverage_percentiles.png"), dpi=130, bbox_inches="tight")
    with open(os.path.join(args.out, "summary_grid_coverage.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
