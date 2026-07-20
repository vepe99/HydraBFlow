"""POSTERIOR-predictive checks for the Ibata (2023) ancillary potential observables.

Companion to ``scripts/ppc_rotation_curve_ibata.py`` (rotation curve) and
``scripts/ppc_ancillary_observables.py`` (the PRIOR version that reads stored npz curves). This one
reuses the draws saved by an ``evaluate_real composition=global`` run -- ``posterior.npz`` (pooled
global = the "composite" potential) and ``single_stream_posterior.npz`` (per member) -- with NO
network re-sampling, rebuilds each draw's potential exactly as the simulator does (imported
``_host_potential`` / ``_stellar_disk_potential`` + the m200_c halo conversion), and pushes it
through the same helpers the simulator uses for the three ancillary observables:

  (1) HI terminal velocity v_term(l)  -- host potential, tangent-point method, vs the observed
      first-quadrant curve (assets/terminal_velocity.csv, McClure-Griffiths & Dickey 2016).
  (2) local surface density Sigma(1.1 kpc) -- host potential, vertical force, vs 71 +/- 6 Msun/pc^2.
  (3) vertical stellar-density profile rho(z) -- STELLAR-disk-only potential, SHAPE (free
      normalization, anchored at z0); no real overlay yet (Ibata 2017b Fig 12f = digitize TODO).

CPU only (forces JAX_PLATFORMS=cpu / empty CUDA_VISIBLE_DEVICES before importing hydrabflow), so it
never touches a GPU. agama does the force/density evals.

Usage:
    uv run python scripts/ppc_ancillary_posterior.py \
        --run-dir outputs/ibata_onedisk_grid_m200c/default/eval_real --n-samples 200
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

# Reuse the verified pot_cfg / identity-constant / posterior loaders from the rotation-curve PPC.
import ppc_rotation_curve_ibata as rc

_DEFAULT_TV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "terminal_velocity.csv"
)


def _band(ax, x, rows, color, label, alpha_fill=(0.15, 0.30), log=False, lw=2.0):
    lo95, lo68, med, hi68, hi95 = np.nanpercentile(rows, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(x, lo95, hi95, color=color, alpha=alpha_fill[0], lw=0)
    ax.fill_between(x, lo68, hi68, color=color, alpha=alpha_fill[1], lw=0)
    ax.plot(x, med, color=color, lw=lw, label=label)
    if log:
        ax.set_yscale("log")
    return np.vstack([lo95, hi95])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="evaluate_real composition=global run dir")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tv", default=_DEFAULT_TV, help="observed terminal-velocity CSV")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from hydrabflow.simulators.stream_agama import (
        _agama, _host_potential, _stellar_disk_potential,
    )
    from hydrabflow.simulators import stream_common as sc

    cfg = OmegaConf.load(os.path.join(args.run_dir, ".hydra", "config.yaml"))
    params = OmegaConf.to_container(cfg.simulator.params, resolve=True)
    pot_cfg = rc._build_pot_cfg(params)
    consts = rc._identity_constants(params["priors_global"])
    names = set(params.get("ancillary_observables", []))
    if not names:
        raise SystemExit("This run has no ancillary_observables in its config.")

    target = {str(k): int(v) for k, v in params["target_streams"].items()}
    idx_to_name = {v: k for k, v in target.items()}

    log10_keys = rc._log10_keys(OmegaConf.to_container(cfg, resolve=True))
    global_post = rc._load_posterior(os.path.join(args.run_dir, "posterior.npz"), log10_keys)
    stream_post = rc._load_posterior(
        os.path.join(args.run_dir, "single_stream_posterior.npz"), log10_keys
    )
    post_keys = list(global_post.keys())
    n_draws = global_post[post_keys[0]].shape[1]
    n = min(args.n_samples, n_draws)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_draws, size=n, replace=False)

    def rows(post, group):
        return [{**consts, **{k: float(post[k][group, i]) for k in post_keys}} for i in idx]

    groups = [("Combined", rows(global_post, 0))]
    for gi in range(stream_post[post_keys[0]].shape[0]):
        groups.append((idx_to_name.get(gi, f"stream{gi}"), rows(stream_post, gi)))

    l_deg = sc.VTERM_L_DEG
    z_kpc = sc.RHO_Z_KPC

    print("Ancillary POSTERIOR-predictive check (Ibata/m200_c) | reusing saved posterior draws")
    print(f"observables: {sorted(names)} | {n} draws x {len(groups)} groups")

    agama = _agama()
    agama.setNumThreads(1)

    # Compute each group's stacks -------------------------------------------------------------
    vterm_c, sigma_c, rho_c = {}, {}, {}
    for name, grows in groups:
        vt = np.full((len(grows), l_deg.size), np.nan)
        sg = np.full(len(grows), np.nan)
        rz = np.full((len(grows), z_kpc.size), np.nan)
        for i, p in enumerate(grows):
            try:
                pot_host = _host_potential(agama, p, pot_cfg)
                if "vterm" in names:
                    vt[i] = sc.terminal_velocity(pot_host, l_deg)
                if "sigma_z" in names:
                    sg[i] = sc.surface_density(pot_host)
                if "rho_z" in names:
                    rz[i] = sc.vertical_density_profile(_stellar_disk_potential(agama, p, pot_cfg), z_kpc)
            except Exception:
                pass
        vterm_c[name], sigma_c[name], rho_c[name] = vt, sg, rz

    # Observed terminal-velocity curve
    tv = None
    try:
        import io
        body = "".join(ln for ln in open(args.tv) if not ln.lstrip().startswith("#"))
        tv = np.genfromtxt(io.StringIO(body), delimiter=",", names=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not read observed terminal-velocity CSV: {exc})")

    summary = {"run_dir": args.run_dir, "n_samples": n, "groups": {}}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Combined": "k", "Pal5": "#d7191c", "NGC3201": "#2c7bb6", "M68": "#fdae61"}
    panels = [nm for nm in ("vterm", "sigma_z", "rho_z") if nm in names]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.8), squeeze=False)
    axes = axes[0]
    ax_i = 0

    # (1) v_term(l): overlay every group's band ----------------------------------------------
    if "vterm" in names:
        ax = axes[ax_i]; ax_i += 1
        for name, _ in groups:
            band = _band(ax, l_deg, vterm_c[name], colors.get(name, "purple"), name,
                         alpha_fill=(0.08, 0.18), lw=1.6)
            if tv is not None:
                lo = np.interp(tv["l_deg"], l_deg, band[0])
                hi = np.interp(tv["l_deg"], l_deg, band[1])
                frac = float(np.mean((tv["vterm_kms"] >= lo) & (tv["vterm_kms"] <= hi)))
                summary["groups"].setdefault(name, {})["vterm_frac_obs_in_95"] = frac
        if tv is not None:
            ax.errorbar(tv["l_deg"], tv["vterm_kms"], yerr=tv["sigma_kms"], fmt="o", ms=4,
                        color="k", capsize=2, label="observed (MGD16)", zorder=6)
        ax.set_xlabel("Galactic longitude l [deg]"); ax.set_ylabel(r"$v_{\rm term}$ [km/s]")
        ax.set_title("HI terminal velocity"); ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # (2) Sigma(1.1 kpc): per-group violins vs 71 +/- 6 ---------------------------------------
    if "sigma_z" in names:
        ax = axes[ax_i]; ax_i += 1
        gnames = [g[0] for g in groups]
        data = [sigma_c[g][np.isfinite(sigma_c[g])] for g in gnames]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for b, g in zip(parts["bodies"], gnames):
            b.set_facecolor(colors.get(g, "purple")); b.set_alpha(0.5)
        ax.axhline(sc.SIGMA_Z_OBS_MSUN_PC2, color="k", lw=2, label="obs 71")
        ax.axhspan(sc.SIGMA_Z_OBS_MSUN_PC2 - sc.SIGMA_Z_ERR_MSUN_PC2,
                   sc.SIGMA_Z_OBS_MSUN_PC2 + sc.SIGMA_Z_ERR_MSUN_PC2, color="k", alpha=0.15)
        ax.set_xticks(range(1, len(gnames) + 1)); ax.set_xticklabels(gnames, rotation=20)
        ax.set_ylabel(r"$\Sigma(1.1\,{\rm kpc})$ [M$_\odot$/pc$^2$]")
        ax.set_title("Local surface density"); ax.legend(fontsize=7); ax.grid(alpha=0.2)
        for g in gnames:
            v = sigma_c[g][np.isfinite(sigma_c[g])]
            summary["groups"].setdefault(g, {})
            summary["groups"][g]["sigma_z_median"] = float(np.median(v))
            summary["groups"][g]["sigma_z_frac_within_obs_1sigma"] = float(
                np.mean(np.abs(v - sc.SIGMA_Z_OBS_MSUN_PC2) <= sc.SIGMA_Z_ERR_MSUN_PC2))

    # (3) rho(z) shape (no real overlay) ------------------------------------------------------
    if "rho_z" in names:
        ax = axes[ax_i]; ax_i += 1
        for name, _ in groups:
            rz = rho_c[name]
            shape = rz / rz[:, [0]]
            _band(ax, z_kpc, shape, colors.get(name, "purple"), name, alpha_fill=(0.08, 0.18),
                  log=True, lw=1.6)
        ax.set_xlabel("z [kpc]"); ax.set_ylabel(r"$\rho(z)/\rho(z_0)$")
        ax.set_title("Vertical stellar density (shape; real=TODO)"); ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    fig.suptitle(f"Ancillary posterior-predictive check ({n} draws/group, reused posterior)", y=1.02)
    fig.tight_layout()
    out = args.out or os.path.join(args.run_dir, "ppc_ancillary_posterior.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    summary["figure"] = out

    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.run_dir, "ppc_ancillary_posterior_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
