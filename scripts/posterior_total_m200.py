"""Total M200 / R200 of the full Milky Way potential for every draw of a global posterior.

The inferred ``log10_M200_TwoPowerTriaxial_halo`` is the DARK HALO alone inside its own r200.
Here R200 is re-solved on the whole potential of each draw (bulge + gas + stellar disk + halo,
the same ``_host_potential`` the simulator used): the radius where the mean enclosed density is
``Delta_mass * rho_crit`` (the run's own cosmology, McMillan H0=70.4, Delta=200), and
M200 = M(<R200). The halo-only solve is reported alongside as a check (it must reproduce the
posterior's log10_M200 up to the 1000 kpc cutoff taper).

Usage: python scripts/posterior_total_m200.py --run-dir <evaluate_real composition=global dir>
Writes <run-dir>/total_m200.{npz,png}.
"""

import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
from ppc_rotation_curve_ibata import (  # noqa: E402
    _build_pot_cfg, _identity_constants, _load_posterior, _log10_keys,
)


def r200_of(pot, rho_target, lo=10.0, hi=2000.0):
    f = lambda r: pot.enclosedMass(r) - rho_target * 4.0 / 3.0 * np.pi * r**3  # noqa: E731
    r = brentq(f, lo, hi, xtol=1e-6)
    return r, float(pot.enclosedMass(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=0, help="0 = all draws")
    ap.add_argument("--recompute", action="store_true", help="ignore a saved total_m200.npz")
    args = ap.parse_args()
    cache = os.path.join(args.run_dir, "total_m200.npz")
    if os.path.exists(cache) and not args.recompute:
        plot(dict(np.load(cache)), args.run_dir)
        return

    from omegaconf import OmegaConf

    from hydrabflow.simulators.stream_agama import (
        _agama, _halo_params_m200c, _host_potential, _resolve_pot_cfg,
    )

    cfg = OmegaConf.load(os.path.join(args.run_dir, ".hydra", "config.yaml"))
    params = OmegaConf.to_container(cfg.simulator.params, resolve=True)
    pot_cfg = _resolve_pot_cfg(_build_pot_cfg(params))
    assert pot_cfg["halo_parameterization"] == "m200_c", "script assumes the m200_c halo"
    consts = _identity_constants(params["priors_global"])
    post = _load_posterior(os.path.join(args.run_dir, "posterior.npz"),
                           _log10_keys(OmegaConf.to_container(cfg, resolve=True)))
    n = post[next(iter(post))].shape[1]
    n = min(n, args.n_samples) if args.n_samples else n

    agama = _agama()
    H0 = float(pot_cfg["halo_H0_kms_mpc"]) / 1000.0
    rho_t = float(pot_cfg["halo_Delta_mass"]) * 3.0 * H0**2 / (8.0 * np.pi * agama.G)

    out = {k: np.full(n, np.nan) for k in ("R200_total", "M200_total", "R200_halo", "M200_halo")}
    for i in range(n):
        row = {**consts, **{k: float(post[k][0, i]) for k in post}}
        out["R200_total"][i], out["M200_total"][i] = r200_of(_host_potential(agama, row, pot_cfg),
                                                             rho_t)
        out["R200_halo"][i], out["M200_halo"][i] = r200_of(
            agama.Potential(_halo_params_m200c(agama, row, pot_cfg)), rho_t)
    out["M200_halo_param"] = 10.0 ** post["log10_M200_TwoPowerTriaxial_halo"][0, :n]
    np.savez(cache, **out)

    dev = np.nanmax(np.abs(out["M200_halo"] / out["M200_halo_param"] - 1))
    print(f"{n} draws | halo-only check: max |M200_halo/param - 1| = {dev:.2e}")
    for k in ("M200_total", "M200_halo", "R200_total", "R200_halo"):
        lo, med, hi = np.nanpercentile(out[k], [16, 50, 84])
        print(f"  {k:11s} {med:.4g} [{lo:.4g}, {hi:.4g}]")
    frac = out["M200_total"] / out["M200_halo_param"] - 1
    print(f"  baryons add {np.median(frac):.1%} to M200 (median)")

    plot(out, args.run_dir)


def plot(out, run_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 20})
    fig, (ax_m, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    for ax, key, xl in [(ax_m, "M200", r"$M_{200}$ [$10^{12}\,M_\odot$]"),
                        (ax_r, "R200", r"$R_{200}$ [kpc]")]:
        tot = out[f"{key}_total"] / (1e12 if key == "M200" else 1.0)
        bins = np.linspace(*np.nanpercentile(tot, [0.5, 99.5]), 50)
        ax.hist(tot, bins, density=True, color=plt.cm.RdYlBu_r(0.0), alpha=0.4)  # compositional blue
        from scipy.stats import gaussian_kde
        x = np.linspace(bins[0], bins[-1], 400)
        ax.plot(x, gaussian_kde(tot[np.isfinite(tot)])(x), color=plt.cm.RdYlBu_r(0.0), lw=2.5)
        lo, med, hi = np.nanpercentile(tot, [16, 50, 84])
        ax.axvline(med, color="k", lw=2)
        for v in (lo, hi):
            ax.axvline(v, color="k", ls="--", lw=1.5)
        ax.set_xlabel(xl)
        ax.set_box_aspect(1)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax_m.set_ylabel("Normalized counts")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(run_dir, f"total_m200.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
