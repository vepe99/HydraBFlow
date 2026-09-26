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
    grid = str(params.get("obs_r_grid", ""))
    extended = grid == "extended"
    if extended:
        obs_r, obs_vc, obs_sig = sc.extended_rotation_curve(split)
    elif grid == "custom":  # explicit config table (v4: Ou et al. 2024)
        obs_r, obs_vc, obs_sig = (np.asarray(params[k], float)
                                  for k in ("obs_r_kpc", "obs_vc_kms", "obs_sigma_vc"))
    else:
        obs_r, obs_vc, obs_sig = sc.OBS_R_KPC, sc.OBS_VC_KMS, sc.OBS_SIGMA_VC
    is_huang = (obs_r > split) if extended else np.zeros(obs_r.size, bool)

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
    # curve at the per-parameter posterior MEDIAN (all saved draws), one per group
    med_rows = [{**consts, **{k: float(np.median(global_post[k][0])) for k in post_keys}}]
    for gi in range(stream_post[post_keys[0]].shape[0]):
        med_rows.append({**consts, **{k: float(np.median(stream_post[k][gi])) for k in post_keys}})

    print("Rotation-curve PPC (Ibata/m200_c potential) | reusing saved posterior draws (NO re-sampling)")
    print(f"halo_parameterization={pot_cfg.get('halo_parameterization', 'rho_a')} | "
          f"gas_disks={pot_cfg.get('gas_disks')} thick_disk={pot_cfg.get('thick_disk')} "
          f"disk_vertical={pot_cfg.get('disk_vertical')} r_t={pot_cfg.get('halo_r_t_kpc')} kpc")
    print(f"grid: {grid or 'Zhou'} ({obs_r.size} radii, "
          f"split {split} kpc) | {n} draws x {len(groups)} groups = {n * len(groups)} curves")

    agama = _agama()
    agama.setNumThreads(1)

    # dense radii for the overlay figure (1 kpc -> just past the outermost Huang 2016 point)
    r_dense = np.geomspace(1.0, 1.05 * sc.HUANG_R_KPC.max(), 120)
    radii = np.concatenate([obs_r, r_dense])

    def vcirc_stack(group_rows):
        out = np.full((len(group_rows), radii.size), np.nan)
        for i, p in enumerate(group_rows):
            try:
                out[i] = _vcirc(_host_potential(agama, p, pot_cfg), radii)
            except Exception:
                pass
        return out[:, :obs_r.size], out[:, obs_r.size:]

    curves, med_curves, dense = {}, {}, {}
    for (name, group_rows), mrow in zip(groups, med_rows):
        vc, dense[name] = vcirc_stack(group_rows)
        curves[name] = vc
        med_curves[name] = vcirc_stack([mrow])[0][0]
        fdev_m = np.nanmedian(np.abs(med_curves[name] - obs_vc) / obs_vc)
        chi2_m = np.nansum(((med_curves[name] - obs_vc) / obs_sig) ** 2)
        print(f"  {name:10s}: curve at posterior-median params: median |frac dev| {fdev_m:5.1%}, "
              f"chi2 {chi2_m:.1f} / {obs_r.size} pts")
        valid = np.isfinite(vc).all(axis=1)
        med = np.nanmedian(vc, axis=0)
        lo, hi = np.nanpercentile(vc, [2.5, 97.5], axis=0)
        cover = np.mean((obs_vc >= lo) & (obs_vc <= hi))
        fdev = np.nanmedian(np.abs(med - obs_vc) / obs_vc)
        print(f"  {name:10s}: {valid.sum():3d}/{n} finite curves | "
              f"median |frac dev| {fdev:5.1%} | 95% band covers {cover:5.1%} of obs points")

    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # same palette as evaluate_real's global-vs-streams corner: RdYlBu_r over [pooled, streams in j order]
    colors = dict(zip([g for g, _ in groups], plt.cm.RdYlBu_r(np.linspace(0, 1, len(groups)))))
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
        ax.plot(r, med_curves[name][order], color=c, lw=1.4, ls="--", label="median params")
        obs_lbl = [(~is_huang, "o", "Ou 2024" if grid == "custom" else "Zhou 2023"),
                   (is_huang, "s", "Huang 2016")]
        for mask, marker, lbl in obs_lbl:
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

    # Overlay figure: one broken r-axis, linear 1-R_BREAK kpc | log R_BREAK-100 kpc, all groups
    # together, the run's own observed curve plus Huang et al. (2016) out to ~100 kpc.
    r_break = 30.0
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.5, 5.8), sharey=True,
                                   gridspec_kw={"wspace": 0, "width_ratios": [1.3, 1]})
    own_lbl = "Ou 2024" if grid == "custom" else "Zhou 2023"
    own = ~is_huang
    for ax in (axl, axr):
        for name, _ in groups:
            vc, c = dense[name], colors.get(name, "purple")
            lo, hi = np.nanpercentile(vc, [16, 84], axis=0)
            ax.fill_between(r_dense, lo, hi, color=c, alpha=0.25, lw=0)
            ax.plot(r_dense, np.nanmedian(vc, axis=0), color=c, lw=1.6, label=f"{'Global' if name == 'Combined' else name} (median, 68%)")
        ax.errorbar(obs_r[own], obs_vc[own], yerr=obs_sig[own], fmt="o", ms=3.5, color="0.1",
                    ecolor="0.45", elinewidth=0.8, capsize=1.5, lw=0, label=own_lbl, zorder=5)
        hu = sc.HUANG_R_KPC > 25.0  # Huang 2016 only beyond 25 kpc
        ax.errorbar(sc.HUANG_R_KPC[hu], sc.HUANG_VC_KMS[hu], yerr=sc.HUANG_SIGMA_VC[hu], fmt="s", ms=3.5,
                    mfc="white", color="0.3", ecolor="0.6", elinewidth=0.8, capsize=1.5, lw=0,
                    label="Huang 2016", zorder=4)
        ax.grid(alpha=0.2)
    axl.set_xlim(1.0, r_break)
    axr.set_xscale("log")
    axr.set_xlim(r_break, r_dense.max())
    axr.set_xticks([40, 50, 70, 100])
    axr.set_xticks([], minor=True)
    axr.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axl.spines["right"].set_visible(False)
    axl.axvline(r_break, color="0.4", ls="--", lw=1, clip_on=False, zorder=6)  # linear|log join
    axr.spines["left"].set_visible(False)
    axr.tick_params(axis="y", which="both", left=False)
    axl.set_xlabel("r [kpc] (linear)")
    axr.set_xlabel("r [kpc] (log)")
    axl.set_ylabel(r"$v_\mathrm{circ}$ [km/s]")
    axl.legend(fontsize=8, loc="lower center", framealpha=0.9, ncol=2)
    fig.suptitle(f"Rotation-curve PPC ({n} posterior draws/group)")
    out2 = os.path.splitext(out)[0] + "_overlay.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
