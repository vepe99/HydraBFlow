"""Prior-predictive coverage of the real Gaia streams in the RAW-PARTICLE representation.

Model-free, CPU-only twin of ``ppc_summary_grid_coverage.py`` for the star-cloud input the
``masked_set_transformer`` arm consumes: the simulated groups go through the TRAINING observation
model (``augment_sim``: window -> member-count subsample -> compact -> Gaia magnitudes -> DR3 errors
-> noise -> v_los mask) and the real members through their preset (``--real-aug``), both are
projected into each stream's data-driven great-circle frame (fitted on the REAL members), and for
each stream we report:

* **quantile envelopes** per observable (phi1, phi2, parallax, mu_phi1, mu_phi2, v_los of MEASURED
  stars only): the real cloud's quantiles against the median and 5-95 % band of the simulated
  clouds' quantiles;
* **2-D overlays** (phi1 vs phi2 / mu_phi1 / mu_phi2 / v_los): pooled simulated star density as
  contours, real members as points;
* a **kernel MMD coverage test**: unbiased RBF MMD^2 between the real cloud and each simulated
  cloud (median-heuristic bandwidth on the standardized real cloud), ranked against the
  sim-vs-sim MMD^2 distribution of random simulated pairs. ``percentile`` = fraction of sim-vs-sim
  values below the MEDIAN real-vs-sim value; ~100 means the real stream is farther from every
  simulation than simulations are from each other (out of distribution). Reported for all six
  observables and for the sky-only / proper-motion-only / v_los-only subsets, to localize WHICH
  observables carry the mismatch.

Outputs (``--out`` dir): ``particle_coverage_<stream>.png`` (quantile panels + 2-D overlays),
``particle_mmd.png`` (MMD ranks), ``particle_coverage.json``.

Usage:
  python scripts/ppc_particle_coverage.py --sim <grouped multistream npz> --out <dir> \
      [--simulator stream_gala_spray_mw22] [--aug stream_global] [--real-aug stream_real_global] [--n-sim 60]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import NAMES, augment_sim, fit_frame, project  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OBS = ["phi1", "phi2", "parallax", "mu_phi1", "mu_phi2", "v_los"]
UNITS = ["deg", "deg", "mas", "mas/yr", "mas/yr", "km/s"]
SUBSETS = {"all": [0, 1, 2, 3, 4, 5], "sky": [0, 1], "pm": [3, 4], "vlos": [5], "parallax": [2]}
QS = np.linspace(0.05, 0.95, 19)
# fixed roles (dataviz: color follows the entity): real = orange, simulations = blue / grey band
C_REAL, C_SIM, C_BAND, C_INK = "#eb6834", "#2a78d6", "#c9d7ea", "#52514e"


def compose_aug(simulator, aug_preset):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from hydrabflow.config import register_configs

    register_configs()
    conf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config",
                      overrides=[f"simulator={simulator}", f"augmentation={aug_preset}", "composition=global"])
    aug = OmegaConf.create(OmegaConf.to_container(cfg.augmentation, resolve=True))
    repo = os.path.dirname(conf_dir)
    res = str(aug.params.get("resources_dir", "data"))
    if not os.path.exists(os.path.join(res, str(aug.params.get("member_table", "apjad382dt1_mrt.txt")))):
        aug.params["resources_dir"] = os.path.join(repo, "assets", "gaia")
    return aug


def real_clouds(real_path, simulator, real_aug, max_particles, seed):
    """Real members through their preset up to the v_los imputation (so the star table is exactly
    what the real-data evaluation feeds the network). Returns {j: (stars (N,6), vlos_mask (N,))}."""
    from omegaconf import OmegaConf

    from hydrabflow.registry import build_augmentations

    d = np.load(real_path)
    m = int(np.asarray(d["j"]).size)
    rb = {}
    for k in d.files:
        if k == "j":
            continue
        a = np.asarray(d[k], dtype=np.float32)
        a = a[:, :max_particles] if a.ndim == 2 else a[:, :, :max_particles]
        if a.ndim >= 3 and a.shape[0] == 1 and a.shape[1] == m:
            a = a.reshape(m, *a.shape[2:])
        if k in ("attention_mask", "vlos_mask") and a.ndim == 2:
            a = a[:, None, :]
        rb[k] = a
    rb["j"] = np.asarray(d["j"]).reshape(m, 1).astype(np.float32)
    aug = compose_aug(simulator, real_aug)
    steps = [str(s) for s in aug.steps]
    keep = [s for s in steps if s in ("sample_obs_error", "impute_vlos")]
    aug = OmegaConf.create(OmegaConf.to_container(aug, resolve=True))
    aug.steps = keep
    for fn in build_augmentations(aug, np.random.default_rng(seed), context={}):
        rb = fn(rb)
    out = {}
    real_clouds.sigma = {}                   # {j: (N,6) sigmas the preset sampled} -- for catalogues without obs_error
    for row in range(m):
        att = np.asarray(rb["attention_mask"])[row, 0].astype(bool)
        vm = np.asarray(rb["vlos_mask"])[row, 0].astype(bool)
        out[int(rb["j"][row, 0])] = (np.asarray(rb["sim_data_projected"])[row][att], vm[att])
        if "sigma_errors" in rb:
            real_clouds.sigma[int(rb["j"][row, 0])] = np.asarray(rb["sigma_errors"])[row][att]
    return out


def to_frame(R, stars):
    """(N, 6) ICRS table -> (N, 6) [phi1, phi2, parallax, mu_phi1, mu_phi2, v_los]."""
    phi1, phi2, mu1, mu2 = project(R, stars[:, 0], stars[:, 1], stars[:, 3], stars[:, 4])
    return np.column_stack([phi1, phi2, stars[:, 2], mu1, mu2, stars[:, 5]])


def mmd2(X, Y, gamma):
    """Unbiased RBF MMD^2 estimator."""
    def k(A, B):
        d2 = np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d2, 0.0))
    kxx, kyy, kxy = k(X, X), k(Y, Y), k(X, Y)
    n, m = len(X), len(Y)
    return ((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
            - 2 * kxy.mean())


def features(F, vm, cols):
    """Rows with every selected observable measured (v_los only where the mask says measured)."""
    sel = np.isfinite(F[:, cols]).all(1)
    if 5 in cols:
        sel &= vm
    return F[sel][:, cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="grouped multistream npz (N,S,P,6)")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--simulator", default="stream_gala_spray_mw22")
    ap.add_argument("--aug", default="stream_global")
    ap.add_argument("--real-aug", default="stream_real_global")
    ap.add_argument("--n-sim", type=int, default=60, help="simulated groups used (random subset)")
    ap.add_argument("--n-null", type=int, default=300, help="sim-vs-sim MMD pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sd = np.load(args.sim)
    x = sd["sim_data_projected"]
    n = x.shape[0]
    pick = np.sort(rng.choice(n, size=min(args.n_sim, n), replace=False))
    sim, attn, vmask = augment_sim(x[pick], np.asarray(sd["j"]).reshape(n, -1)[pick],
                                   aug_preset=args.aug, simulator=args.simulator, seed=args.seed)
    aug = compose_aug(args.simulator, args.aug)
    real = real_clouds(args.real, args.simulator, args.real_aug, int(aug.params.get("max_particles", 300)), args.seed)
    jj = np.asarray(sd["j"]).reshape(n, -1)[pick].astype(int)

    report = {"sim": args.sim, "n_groups_used": int(len(pick)), "streams": {}}
    mmd_summary = {}
    for j, name in NAMES.items():
        if j not in real:
            continue
        r_stars, r_vm = real[j]
        R = fit_frame(r_stars[:, 0], r_stars[:, 1])
        Fr = to_frame(R, r_stars)
        clouds = []
        for g in range(len(pick)):
            s = int(np.flatnonzero(jj[g] == j)[0])
            a = attn[g, s]
            if a.sum() < 20:
                continue
            clouds.append((to_frame(R, sim[g, s][a]), vmask[g, s][a]))
        if len(clouds) < 5:
            print(f"{name}: only {len(clouds)} usable simulated clouds — skipped")
            continue

        # --- quantile envelopes
        qr = np.full((6, len(QS)), np.nan)
        qs = np.full((len(clouds), 6, len(QS)), np.nan)
        for c in range(6):
            v = Fr[:, c] if c != 5 else Fr[r_vm, c]
            if v.size > 3:
                qr[c] = np.quantile(v, QS)
            for gi, (F, vm) in enumerate(clouds):
                v = F[:, c] if c != 5 else F[vm, c]
                if v.size > 3:
                    qs[gi, c] = np.quantile(v, QS)
        qlo, qmed, qhi = np.nanpercentile(qs, [5, 50, 95], axis=0)
        # fraction of real quantiles inside the 5-95 % envelope, per observable
        inside = {OBS[c]: float(np.mean((qr[c] >= qlo[c]) & (qr[c] <= qhi[c]))) for c in range(6)}

        # --- MMD coverage per observable subset
        mm = {}
        for sub, cols in SUBSETS.items():
            Xr = features(Fr, r_vm, cols)
            if len(Xr) < 10:
                continue
            mu, sc = np.median(Xr, 0), 1.4826 * np.median(np.abs(Xr - np.median(Xr, 0)), 0) + 1e-9
            Zr = (Xr - mu) / sc
            d2 = np.sum((Zr[:, None] - Zr[None]) ** 2, -1)
            gamma = 1.0 / (2 * np.median(d2[d2 > 0]))
            Zs = [(features(F, vm, cols) - mu) / sc for F, vm in clouds]
            Zs = [Z for Z in Zs if len(Z) >= 10]
            if len(Zs) < 5:
                continue
            d_real = np.array([mmd2(Zr, Z, gamma) for Z in Zs])
            pairs = rng.integers(0, len(Zs), size=(args.n_null, 2))
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            d_null = np.array([mmd2(Zs[a], Zs[b], gamma) for a, b in pairs])
            mm[sub] = dict(real_vs_sim_median=float(np.median(d_real)),
                           real_vs_sim_p5=float(np.percentile(d_real, 5)),
                           sim_vs_sim_median=float(np.median(d_null)),
                           sim_vs_sim_p95=float(np.percentile(d_null, 95)),
                           percentile=float(100 * np.mean(d_null <= np.median(d_real))),
                           frac_sims_farther_than_typical=float(np.mean(d_real > np.percentile(d_null, 95))))
        mmd_summary[name] = mm
        report["streams"][name] = dict(n_real=int(len(Fr)), n_real_vlos=int(r_vm.sum()), n_sim_clouds=len(clouds),
                                       quantile_inside_5_95=inside, mmd=mm,
                                       real_quantiles={OBS[c]: qr[c].tolist() for c in range(6)},
                                       sim_quantile_median={OBS[c]: qmed[c].tolist() for c in range(6)})
        print(f"{name:8s} real {len(Fr)} stars ({int(r_vm.sum())} vlos) vs {len(clouds)} sims | "
              f"quantiles inside 5-95%: " + " ".join(f"{k}={v:.2f}" for k, v in inside.items()) + " | MMD pct: "
              + " ".join(f"{k}={v['percentile']:.0f}" for k, v in mm.items()))

        # --- figure: quantile panels (row 1) + 2-D overlays (row 2)
        fig, axes = plt.subplots(2, 6, figsize=(22, 7.5))
        for c in range(6):
            ax = axes[0, c]
            ax.fill_between(QS, qlo[c], qhi[c], color=C_BAND, lw=0, label="sims 5-95 %")
            ax.plot(QS, qmed[c], color=C_SIM, lw=2, label="sims median")
            ax.plot(QS, qr[c], color=C_REAL, lw=2, marker="o", ms=4, label="real Gaia")
            ax.set_xlabel("quantile")
            ax.set_title(f"{OBS[c]} [{UNITS[c]}]  inside {100*inside[OBS[c]]:.0f} %", fontsize=10)
            ax.grid(alpha=0.25, lw=0.5)
            if c == 0:
                ax.legend(fontsize=8, frameon=False)
        pooled = np.concatenate([F for F, _ in clouds])
        pooled_vm = np.concatenate([vm for _, vm in clouds])
        for k, c in enumerate((1, 2, 3, 4, 5)):
            ax = axes[1, k]
            P = pooled if c != 5 else pooled[pooled_vm]
            Rr = Fr if c != 5 else Fr[r_vm]
            lo = np.nanpercentile(np.concatenate([P[:, c], Rr[:, c]]), 1)
            hi = np.nanpercentile(np.concatenate([P[:, c], Rr[:, c]]), 99)
            x0, x1 = np.nanpercentile(np.concatenate([P[:, 0], Rr[:, 0]]), [0.5, 99.5])
            H, xe, ye = np.histogram2d(P[:, 0], P[:, c], bins=[40, 40], range=[[x0, x1], [lo, hi]])
            H = H / max(H.max(), 1)
            ax.contourf((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, H.T,
                        levels=[0.02, 0.1, 0.3, 0.6, 1.0], colors=["#e6eef8", C_BAND, "#9dbbe0", "#6d9bd3", C_SIM], alpha=0.9)
            ax.scatter(Rr[:, 0], Rr[:, c], s=7, color=C_REAL, lw=0, label="real Gaia")
            ax.set_xlabel("phi1 [deg]")
            ax.set_ylabel(f"{OBS[c]} [{UNITS[c]}]")
            ax.set_ylim(lo, hi)
            if k == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper right")
        axes[1, 5].axis("off")
        axes[1, 5].text(0.0, 0.95, "MMD coverage (percentile of the median\nreal-vs-sim MMD² in the sim-vs-sim null):\n\n"
                        + "\n".join(f"{k:9s} {v['percentile']:5.1f}" for k, v in mm.items()),
                        va="top", ha="left", fontsize=10, family="monospace", color=C_INK, transform=axes[1, 5].transAxes)
        fig.suptitle(f"{name}: raw-particle prior-predictive coverage — {len(clouds)} simulated streams "
                     f"(training observation model) vs the real Gaia members. {args.title}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"particle_coverage_{name}.png"), dpi=120)
        plt.close(fig)

    # --- MMD rank overview
    if mmd_summary:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        names = list(mmd_summary)
        subs = list(SUBSETS)
        w = 0.8 / len(subs)
        greys = ["#0b0b0b", "#52514e", "#8a8983", "#b5b4ad", "#d6d5cf"]
        for si, sub in enumerate(subs):
            vals = [mmd_summary[nm].get(sub, {}).get("percentile", np.nan) for nm in names]
            ax.bar(np.arange(len(names)) + (si - len(subs) / 2 + 0.5) * w, vals, width=w * 0.92,
                   color=greys[si], label=sub)
        ax.axhline(95, color=C_REAL, lw=1, ls="--")
        ax.text(len(names) - 0.5, 96, "95th", color=C_REAL, fontsize=8, ha="right")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylim(0, 105)
        ax.set_ylabel("MMD percentile")
        ax.legend(fontsize=8, frameon=False, ncol=len(subs), loc="lower left")
        ax.set_title("Raw-particle MMD coverage by observable subset (100 = real farther from every sim than sims from each other)", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "particle_mmd.png"), dpi=130)
        plt.close(fig)

    with open(os.path.join(args.out, "particle_coverage.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
