"""Posterior-predictive check on the *model rotation curve* only.

Takes a completed ``evaluate_real composition=global`` run directory (which saved both the pooled
global posterior ``posterior.npz`` and the per-member ``single_stream_posterior.npz`` -- the very
draws used for the ``real_global_vs_streams_corner.png`` overlay). It does NOT re-run the network:
it reuses those already-sampled parameter draws, subsamples ``--n-samples`` of them per group,
builds the host potential for each draw and evaluates its circular-velocity curve on the observed
rotation-curve grid, then overlays the resulting posterior-predictive bands on the observed
Zhou (2023) u Huang (2016) rotation curve.

This isolates the rotation-curve observable: it answers "does the potential the network inferred
(per single stream, and pooled) reproduce the measured Milky Way rotation curve?" -- independent of
the stream-track observable.

Standalone: imports only agama + numpy + matplotlib (+ omegaconf to read the run config). No
hydrabflow / BayesFlow / JAX import, so it runs on a CPU-only cluster and starts instantly.

Usage:
    uv run python scripts/ppc_rotation_curve.py \
        --run-dir outputs/stream_agama/stream_fusion_model5_spray_huang/2026-07-08_14-42-49 \
        --n-samples 100
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# --------------------------------------------------------------------------------------------- #
# Host potential + circular velocity, copied verbatim from hydrabflow.simulators.stream_agama /
# stream_common so this PPC is identical to what the simulator's rejection prior computes, without
# importing the (JAX-heavy) hydrabflow package.
# --------------------------------------------------------------------------------------------- #
BULGE_PARAMS = dict(
    type="Spheroid", scaleRadius=75 / 1e3, densityNorm=9.6e10, gamma=0, alpha=1, beta=1.8,
    cutoffStrength=2, outerCutoffRadius=2.1, axisRatioY=1.0, axisRatioZ=0.5,
)

# Zhou et al. (2023) rotation curve (radii kpc, Vc km/s, 1-sigma km/s).
OBS_R_KPC = np.array([
    5.24, 5.74, 6.25, 6.77, 7.23, 7.83, 8.21, 8.78, 9.26, 9.75,
    10.25, 10.75, 11.25, 11.75, 12.24, 12.74, 13.25, 13.74, 14.23, 14.74,
    15.23, 15.74, 16.24, 16.74, 17.23, 17.74, 18.35, 18.90, 19.50, 20.41,
    21.28, 22.39, 23.16, 24.00,
])
OBS_VC_KMS = np.array([
    225.10, 233.53, 234.30, 233.17, 236.19, 236.00, 233.19, 233.15, 232.15, 231.24,
    230.34, 230.54, 229.11, 227.48, 226.69, 225.56, 224.90, 223.57, 221.10, 220.19,
    219.59, 217.36, 216.61, 217.28, 216.25, 213.81, 217.53, 212.10, 210.46, 206.69,
    207.71, 203.72, 205.20, 200.64,
])
OBS_SIGMA_VC = np.array([
    0.69, 0.68, 0.62, 0.60, 0.45, 0.29, 0.26, 0.22, 0.17, 0.16,
    0.17, 0.18, 0.19, 0.20, 0.25, 0.27, 0.27, 0.31, 0.40, 0.43,
    0.50, 0.68, 0.74, 0.87, 1.02, 1.15, 1.45, 1.58, 1.32, 1.71,
    1.69, 2.01, 2.50, 4.94,
])
# Huang et al. (2016) rotation curve to ~100 kpc (radii kpc, Vc km/s, 1-sigma km/s).
_HUANG = np.array([
    [4.60, 231.24, 7.00], [5.08, 230.46, 7.00], [5.58, 230.01, 7.00],
    [6.10, 239.61, 7.00], [6.57, 246.27, 7.00], [7.07, 243.49, 7.00],
    [7.58, 242.71, 7.00], [8.04, 243.23, 7.00],
    [8.34, 239.89, 5.92], [8.65, 237.26, 6.29], [9.20, 235.30, 5.60],
    [9.62, 230.99, 5.49], [10.09, 228.41, 5.62], [10.58, 224.26, 5.87],
    [11.09, 224.94, 7.02], [11.58, 233.57, 7.65], [12.07, 240.02, 6.17],
    [12.73, 242.21, 8.64], [13.72, 261.78, 14.89], [14.95, 259.26, 30.84],
    [15.52, 268.57, 49.67], [16.55, 261.17, 50.91], [17.56, 240.66, 49.91],
    [18.54, 215.31, 24.80], [19.50, 214.99, 24.42], [21.25, 251.68, 19.50],
    [23.78, 259.65, 19.62], [26.22, 242.02, 18.66], [28.71, 224.11, 16.97],
    [31.29, 211.20, 16.43], [33.73, 217.93, 17.66], [36.19, 219.33, 18.44],
    [38.73, 213.31, 17.29], [41.25, 200.05, 17.72], [43.93, 190.15, 18.65],
    [46.43, 198.95, 20.70], [48.71, 192.91, 19.24], [51.56, 198.90, 21.74],
    [57.03, 185.88, 21.56], [62.55, 173.89, 22.87], [69.47, 196.36, 25.89],
    [79.27, 175.05, 22.71], [98.97, 147.72, 23.55],
])
HUANG_R_KPC, HUANG_VC_KMS, HUANG_SIGMA_VC = _HUANG.T


def extended_rotation_curve(split_kpc):
    """Zhou up to split, Huang beyond; sorted by radius. (r, vc, sigma)."""
    split = float(OBS_R_KPC.max()) if split_kpc is None else float(split_kpc)
    hi = HUANG_R_KPC > split
    r = np.concatenate([OBS_R_KPC, HUANG_R_KPC[hi]])
    vc = np.concatenate([OBS_VC_KMS, HUANG_VC_KMS[hi]])
    sig = np.concatenate([OBS_SIGMA_VC, HUANG_SIGMA_VC[hi]])
    order = np.argsort(r)
    return r[order], vc[order], sig[order]


def _agama():
    import agama

    agama.setUnits(length=1, velocity=1, mass=1)
    return agama


def _host_potential(agama, p):
    return agama.Potential(
        BULGE_PARAMS,
        dict(type="Spheroid", scaleRadius=p["a_TwoPowerTriaxial_halo"],
             densityNorm=p["rho_TwoPowerTriaxial_halo"], gamma=p["gamma_TwoPowerTriaxial_halo"],
             alpha=1, beta=p["beta_TwoPowerTriaxial_halo"], cutoffStrength=2,
             outerCutoffRadius=np.inf, axisRatioY=1.0, axisRatioZ=p["q_TwoPowerTriaxial_halo"]),
        dict(type="Disk", scaleRadius=p["r_Disk"], scaleHeight=p["z_Disk"],
             surfaceDensity=p["Sigma_Disk"], sersicIndex=1, innerCutoffRadius=0),
    )


def _vcirc(pot_host, obs_r):
    """Model circular velocity [km/s] at obs_r; NaN where v^2 < 0."""
    points = np.column_stack((obs_r, np.zeros_like(obs_r), np.zeros_like(obs_r)))
    v2 = -obs_r * pot_host.force(points)[:, 0]
    return np.sqrt(np.where(v2 > 0, v2, np.nan))


# Global potential parameters, in the order _host_potential consumes them.
GLOBAL_KEYS = [
    "rho_TwoPowerTriaxial_halo", "gamma_TwoPowerTriaxial_halo", "a_TwoPowerTriaxial_halo",
    "beta_TwoPowerTriaxial_halo", "q_TwoPowerTriaxial_halo", "r_Disk", "z_Disk", "Sigma_Disk",
]


def _load_posterior(path):
    d = np.load(path, allow_pickle=True)
    return {k: np.asarray(d[k]).reshape(d[k].shape[0], -1) for k in d.files}


def _rows(post, group, idx):
    return [{k: float(post[k][group, i]) for k in GLOBAL_KEYS} for i in idx]


def _vcirc_stack(rows, obs_r):
    agama = _agama()
    agama.setNumThreads(1)
    out = np.full((len(rows), obs_r.size), np.nan)
    for i, p in enumerate(rows):
        try:
            out[i] = _vcirc(_host_potential(agama, p), obs_r)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="evaluate_real composition=global run dir")
    ap.add_argument("--n-samples", type=int, default=100,
                    help="posterior draws per group reused from the saved posterior (no re-sampling)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(args.run_dir, ".hydra", "config.yaml"))
    params = cfg.simulator.params
    split = float(params.get("obs_r_split_kpc", float(OBS_R_KPC.max())))
    extended = str(params.get("obs_r_grid", "")) == "extended"
    if extended:
        obs_r, obs_vc, obs_sig = extended_rotation_curve(split)
    else:
        obs_r, obs_vc, obs_sig = OBS_R_KPC, OBS_VC_KMS, OBS_SIGMA_VC
    is_huang = obs_r > split

    target = {str(k): int(v) for k, v in OmegaConf.to_container(params.target_streams).items()}
    idx_to_name = {v: k for k, v in target.items()}

    global_post = _load_posterior(os.path.join(args.run_dir, "posterior.npz"))
    stream_post = _load_posterior(os.path.join(args.run_dir, "single_stream_posterior.npz"))
    n_draws = global_post[GLOBAL_KEYS[0]].shape[1]
    n = min(args.n_samples, n_draws)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_draws, size=n, replace=False)  # reuse saved draws, no network

    groups = [("Combined", _rows(global_post, 0, idx))]
    for gi in range(stream_post[GLOBAL_KEYS[0]].shape[0]):
        groups.append((idx_to_name.get(gi, f"stream{gi}"), _rows(stream_post, gi, idx)))

    print(f"Rotation-curve PPC | reusing saved posterior draws (NO re-sampling)")
    print(f"grid: {'extended Zhou u Huang' if extended else 'Zhou'} ({obs_r.size} radii, "
          f"split {split} kpc) | {n} draws x {len(groups)} groups = {n * len(groups)} curves")

    curves = {}
    for name, rows in groups:
        vc = _vcirc_stack(rows, obs_r)
        curves[name] = vc
        valid = np.isfinite(vc).all(axis=1)
        med = np.nanmedian(vc, axis=0)
        lo, hi = np.nanpercentile(vc, [2.5, 97.5], axis=0)
        cover = np.mean((obs_vc >= lo) & (obs_vc <= hi))
        fdev = np.nanmedian(np.abs(med - obs_vc) / obs_vc)
        print(f"  {name:10s}: {valid.sum():3d}/{n} finite curves | "
              f"median |frac dev| {fdev:5.1%} | 95% band covers {cover:5.1%} of obs points")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Combined": "k", "Pal5": "#d7191c", "NGC3201": "#2c7bb6", "M68": "#fdae61"}
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.2),
                             sharex=True, sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (name, _) in zip(axes, groups):
        vc = curves[name]
        c = colors.get(name, "purple")
        order = np.argsort(obs_r)
        r = obs_r[order]
        med = np.nanmedian(vc, axis=0)[order]
        lo68, hi68 = np.nanpercentile(vc, [16, 84], axis=0)
        lo95, hi95 = np.nanpercentile(vc, [2.5, 97.5], axis=0)
        ax.fill_between(r, lo95[order], hi95[order], color=c, alpha=0.15, lw=0)
        ax.fill_between(r, lo68[order], hi68[order], color=c, alpha=0.30, lw=0)
        ax.plot(r, med, color=c, lw=1.8, label="PPC median")
        for mask, marker, lbl in [(~is_huang, "o", "Zhou 2023"), (is_huang, "s", "Huang 2016")]:
            if mask.any():
                ax.errorbar(obs_r[mask], obs_vc[mask], yerr=obs_sig[mask], fmt=marker, ms=3.5,
                            color="0.15", ecolor="0.55", elinewidth=0.8, capsize=1.5, lw=0,
                            label=lbl, zorder=5)
        if extended:
            ax.axvline(split, color="0.7", ls=":", lw=1)
        ax.set_title(name)
        ax.set_xlabel("r [kpc]")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"$v_\mathrm{circ}$ [km/s]")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Rotation-curve posterior-predictive check "
                 f"({n} draws/group, reused from saved posterior)", y=1.02)
    fig.tight_layout()
    out = args.out or os.path.join(args.run_dir, "ppc_rotation_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
