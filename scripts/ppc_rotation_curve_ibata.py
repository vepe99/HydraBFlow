"""Rotation-curve posterior-predictive check for the Ibata / m200_c potential.

Same idea as ``scripts/ppc_rotation_curve.py`` (reuse the draws saved by an
``evaluate_real composition=global`` run -- ``posterior.npz`` = pooled global, and
``single_stream_posterior.npz`` = per-member -- NO network re-sampling), but it builds the
FULL potential the simulator actually used for this dataset instead of the legacy
bulge+halo+one-disk model. It reads ``simulator.params`` from the run's ``.hydra/config.yaml`` and
assembles ``pot_cfg`` (gas disks, thick disk, halo truncation, exponential vertical profile,
m200_c vs rho_a halo parameterization) exactly as ``AgamaStreamSimulator`` does, then calls the
real ``_host_potential`` / ``_vcirc`` from ``hydrabflow.simulators.stream_agama``.

For the m200_c halo the saved posterior carries ``log10_M200`` / ``ln_cvprime`` (not rho/a); the
imported ``_host_potential`` -> ``_halo_params_m200c`` does the McMillan (2017) conversion per draw,
so this PPC is bit-for-bit consistent with the simulator. Fixed (identity-prior) globals such as
``beta_..._halo`` are read from the config and merged into every draw.

CPU only: forces ``JAX_PLATFORMS=cpu`` / empty ``CUDA_VISIBLE_DEVICES`` before importing hydrabflow,
so it never touches a GPU (e.g. while a tuning job runs). agama does the potential eval.

Usage:
    uv run python scripts/ppc_rotation_curve_ibata.py \
        --run-dir outputs/ibata_onedisk_grid_m200c/default/eval_real \
        --n-samples 200
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

# pot_cfg keys understood by _resolve_pot_cfg / _host_potential, filled from simulator.params.
_POT_CFG_KEYS = [
    "halo_r_t_kpc", "gas_disks", "thick_disk", "disk_vertical", "bulge_density_norm",
    "halo_parameterization", "halo_H0_kms_mpc", "halo_Delta_mass", "halo_Delta_c",
]


def _build_pot_cfg(params) -> dict:
    """Pick the potential-config knobs out of simulator.params (missing keys fall back to the
    _DEFAULT_POT_CFG legacy values inside _resolve_pot_cfg)."""
    cfg = {}
    for k in _POT_CFG_KEYS:
        if k in params and params[k] is not None:
            v = params[k]
            cfg[k] = bool(v) if isinstance(v, bool) else v
    return cfg


def _identity_constants(priors_global) -> dict:
    """Fixed globals (type=identity, e.g. beta_..._halo, or rho/a when unused) as scalars."""
    out = {}
    for name, spec in priors_global.items():
        if str(spec.get("type", "")) == "identity":
            out[str(name)] = float(spec["prior_parameters"][0])
    return out


def _log10_keys(cfg) -> set:
    """Keys the run's preprocessing stores in log10 space. evaluate_real saves posterior.npz /
    single_stream_posterior.npz in the model's NATIVE (preprocessed) space (evaluate_real.py:173),
    so these keys must be inverted (10**x) before feeding agama physical units."""
    keys = set()
    for step in cfg.get("preprocessing", {}).get("steps", []) or []:
        if str(step.get("name", "")) == "log10_transform":
            keys.update(str(k) for k in (step.get("keys") or []))
    return keys


def _load_posterior(path, log10_keys=()):
    d = np.load(path, allow_pickle=True)
    # (group, draw, 1) -> (group, draw); invert the log10 preprocessing back to physical units.
    out = {k: np.asarray(d[k]).reshape(d[k].shape[0], -1) for k in d.files}
    for k in out:
        if k in log10_keys:
            out[k] = 10.0 ** out[k]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="evaluate_real composition=global run dir")
    ap.add_argument("--n-samples", type=int, default=200,
                    help="posterior draws per group reused from the saved posterior (no re-sampling)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from hydrabflow.simulators.stream_agama import _agama, _host_potential, _vcirc
    from hydrabflow.simulators import stream_common as sc

    cfg = OmegaConf.load(os.path.join(args.run_dir, ".hydra", "config.yaml"))
    params = OmegaConf.to_container(cfg.simulator.params, resolve=True)

    pot_cfg = _build_pot_cfg(params)
    consts = _identity_constants(params["priors_global"])

    split = float(params.get("obs_r_split_kpc", float(sc.OBS_R_KPC.max())))
    extended = str(params.get("obs_r_grid", "")) == "extended"
    if extended:
        obs_r, obs_vc, obs_sig = sc.extended_rotation_curve(split)
    else:
        obs_r, obs_vc, obs_sig = sc.OBS_R_KPC, sc.OBS_VC_KMS, sc.OBS_SIGMA_VC
    is_huang = obs_r > split

    target = {str(k): int(v) for k, v in params["target_streams"].items()}
    idx_to_name = {v: k for k, v in target.items()}

    log10_keys = _log10_keys(OmegaConf.to_container(cfg, resolve=True))
    global_post = _load_posterior(os.path.join(args.run_dir, "posterior.npz"), log10_keys)
    stream_post = _load_posterior(
        os.path.join(args.run_dir, "single_stream_posterior.npz"), log10_keys
    )
    post_keys = list(global_post.keys())

    n_draws = global_post[post_keys[0]].shape[1]
    n = min(args.n_samples, n_draws)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_draws, size=n, replace=False)  # reuse saved draws, no network

    def rows(post, group):
        return [{**consts, **{k: float(post[k][group, i]) for k in post_keys}} for i in idx]

    groups = [("Combined", rows(global_post, 0))]
    for gi in range(stream_post[post_keys[0]].shape[0]):
        groups.append((idx_to_name.get(gi, f"stream{gi}"), rows(stream_post, gi)))

    print("Rotation-curve PPC (Ibata/m200_c potential) | reusing saved posterior draws (NO re-sampling)")
    print(f"halo_parameterization={pot_cfg.get('halo_parameterization', 'rho_a')} | "
          f"gas_disks={pot_cfg.get('gas_disks')} thick_disk={pot_cfg.get('thick_disk')} "
          f"disk_vertical={pot_cfg.get('disk_vertical')} r_t={pot_cfg.get('halo_r_t_kpc')} kpc")
    print(f"grid: {'extended Zhou u Huang' if extended else 'Zhou'} ({obs_r.size} radii, "
          f"split {split} kpc) | {n} draws x {len(groups)} groups = {n * len(groups)} curves")

    agama = _agama()
    agama.setNumThreads(1)

    def vcirc_stack(group_rows):
        out = np.full((len(group_rows), obs_r.size), np.nan)
        for i, p in enumerate(group_rows):
            try:
                out[i] = _vcirc(_host_potential(agama, p, pot_cfg), obs_r)
            except Exception:
                pass
        return out

    curves = {}
    for name, group_rows in groups:
        vc = vcirc_stack(group_rows)
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
    order = np.argsort(obs_r)
    r = obs_r[order]
    for ax, (name, _) in zip(axes, groups):
        vc = curves[name]
        c = colors.get(name, "purple")
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
    fig.suptitle(f"Rotation-curve PPC (Ibata/m200_c potential; {n} draws/group, reused posterior)",
                 y=1.02)
    fig.tight_layout()
    out = args.out or os.path.join(args.run_dir, "ppc_rotation_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
