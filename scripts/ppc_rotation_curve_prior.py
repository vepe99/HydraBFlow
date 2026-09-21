"""Prior-predictive check of the ROTATION-CURVE observable against the observed curve.

Reads ``vcirc_kms`` (n, n_r, 1) from a flat or grouped npz, the observed curve + errors from the
simulator config (``obs_r_kpc`` / ``obs_vc_kms`` / ``obs_sigma_vc``, e.g. Ou et al. 2024 for v4), and
reports: the prior band per radius vs the observed points, P(sim < obs) per radius, the chi2
distribution over rows (n_r dof), the fraction of rows within 1/2 sigma at EVERY radius, and the
Spearman correlation of log chi2 with every varied global (which parameters the curve constrains).
Noise: the observed sigma enters the chi2; the training-time ``add_noise_to_vcirc`` draw is not
applied to the sims (it would double-count the same sigma).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--simulator", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r-min", type=float, default=0.0, help="ignore radii below this [kpc]")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from hydra import compose, initialize_config_dir
    from scipy.stats import spearmanr

    from hydrabflow.config import register_configs

    register_configs()
    with initialize_config_dir(config_dir=os.path.abspath("conf"), version_base=None):
        cfg = compose("config", overrides=[f"simulator={args.simulator}"])
    P = cfg.simulator.params
    r = np.asarray(P.obs_r_kpc, float)
    obs = np.asarray(P.obs_vc_kms, float)
    sig = np.asarray(P.obs_sigma_vc, float)
    keep = r >= args.r_min

    ds = np.load(args.sim)
    vc = np.asarray(ds["vcirc_kms"], float)
    vc = vc.reshape(vc.shape[0], -1)  # (n, n_r)
    ok = np.isfinite(vc).all(1)
    vc = vc[ok]
    assert vc.shape[1] == len(r), (vc.shape, len(r))
    n = vc.shape[0]

    res = (vc[:, keep] - obs[keep]) / sig[keep]
    chi2 = (res**2).sum(1)
    dof = int(keep.sum())
    within1 = (np.abs(res) <= 1).all(1)
    within2 = (np.abs(res) <= 2).all(1)
    p_below = (vc[:, keep] < obs[keep]).mean(0)
    band = np.percentile(vc, [5, 16, 50, 84, 95], axis=0)
    frac_dev = np.median(np.abs(vc[:, keep] / obs[keep] - 1), axis=1)

    globs = {}
    nrow = int(np.asarray(ds["j"]).shape[0]) if "j" in ds.files else n
    for k in ds.files:
        a = np.asarray(ds[k])
        if k in ("sim_data_projected", "vcirc_kms", "j") or k.endswith("_derived"):
            continue
        if a.ndim == 2 and a.shape == (nrow, 1) and np.ptp(a) > 0:
            globs[k] = a[:, 0][ok]
    order = np.argsort(chi2)
    rep = {
        "n_rows": int(n), "dof": dof, "radii_kpc": r[keep].tolist(),
        "chi2": {"min": float(chi2.min()), "p5": float(np.percentile(chi2, 5)),
                 "median": float(np.median(chi2)), "frac_below_dof": float((chi2 < dof).mean()),
                 "frac_below_2dof": float((chi2 < 2 * dof).mean())},
        "frac_within_1sigma_all_radii": float(within1.mean()),
        "frac_within_2sigma_all_radii": float(within2.mean()),
        "median_frac_dev_prior_median": float(np.median(frac_dev)),
        "p_sim_below_obs_per_radius": np.round(p_below, 3).tolist(),
        "obs_inside_prior_5_95": bool(((obs >= band[0]) & (obs <= band[4])).all()),
        "obs_inside_prior_16_84_per_radius": ((obs >= band[1]) & (obs <= band[3])).astype(int).tolist(),
        "best_row_params": {k: float(v[order[0]]) for k, v in globs.items()},
        "spearman_logchi2_vs_param": {k: round(float(spearmanr(v, np.log(chi2)).statistic), 3)
                                      for k, v in globs.items()},
    }
    json.dump(rep, open(os.path.join(args.out, "rotation_curve_prior.json"), "w"), indent=1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axs[0]
    ax.fill_between(r, band[0], band[4], color="C0", alpha=0.15, label="prior 5-95 %")
    ax.fill_between(r, band[1], band[3], color="C0", alpha=0.3, label="prior 16-84 %")
    ax.plot(r, band[2], "C0-", label="prior median")
    for i in order[:10]:
        ax.plot(r, vc[i], color="C1", alpha=0.6, lw=0.9)
    ax.plot([], [], color="C1", label="10 best rows (chi2)")
    ax.errorbar(r, obs, yerr=sig, fmt="ko", ms=4, capsize=2, label="observed (config table)")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("v_c [km/s]")
    ax.set_title(f"rotation curve: {n} prior rows, best chi2 {chi2.min():.0f}/{dof} dof")
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(r[keep], p_below, "ko-")
    ax.axhline(0.5, color="0.5", ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("P(sim < obs)")
    ax.set_title("prior percentile of the observed point per radius")
    ax = axs[2]
    ax.hist(np.log10(chi2), bins=60, color="C0")
    ax.axvline(np.log10(dof), color="k", ls="--", label=f"chi2 = dof ({dof})")
    ax.set_xlabel("log10 chi2 vs observed curve")
    ax.set_ylabel("rows")
    ax.set_title(f"within 1 sigma at every radius: {100 * within1.mean():.1f} % of the prior")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "rotation_curve_prior.png"), dpi=130)

    print(json.dumps({k: v for k, v in rep.items() if k not in ("radii_kpc",)}, indent=1))


if __name__ == "__main__":
    main()
