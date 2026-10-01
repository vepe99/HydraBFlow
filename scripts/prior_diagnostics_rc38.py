"""Prior diagnostics for the rc38 simulator (potentials only, no streams).

Draws ``--n`` rows from the rc38 prior, builds each row's host potential exactly as the simulator
does (``_host_potential`` + ``_solar_frame`` + ``_m200c_derived``), and reports:

* the fraction of draws with M* < M_bulge and on the M_disk floor, overall and per log10_M200 bin;
* histograms + quantiles of every SAMPLED parameter (inferred and nuisance) and every derived
  quantity (M_bulge, M_disk, M_thin, M_thick, Sigma_thin, Sigma_thick, R_d_thin, R_d_thick, f_thick,
  M(<50 kpc), v_c(R0), (v_c(R0) + V_sun)/R0);
* the same quantities for the McMillan (2017) best-fit potential (agama's McMillan17.ini, R0 = 8.21,
  V_sun = 12.24), marked on every panel;
* v_c(R0) against log10_M200, with the observed v_c near R0 from the rc38 rotation-curve table.

Outputs ``summary.json`` + PNGs in ``--out``.

Run:  HYDRABFLOW_NUM_GPUS=0 JAX_PLATFORMS=cpu .venv/bin/python scripts/prior_diagnostics_rc38.py \
        --out <dir> --n 20000 --n-workers 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

SIM = "stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38"
INFERRED = ["gamma_TwoPowerTriaxial_halo", "q_TwoPowerTriaxial_halo", "log10_M200_TwoPowerTriaxial_halo",
            "ln_cvprime_TwoPowerTriaxial_halo", "log10_Mstar", "ln_R_d_thin", "ln_R_d_thick", "ln_f_thick"]
NUISANCE = ["R0_Sun", "U_Sun", "V_Sun", "W_Sun"]
DERIVED = ["M_bulge", "M_disk", "M_thin", "M_thick", "Sigma_thin", "Sigma_thick", "R_d_thin",
           "R_d_thick", "f_thick", "M50", "vc_R0", "Omega_sun"]
LOG_AXES = {"M_disk", "M_thin", "M_thick", "Sigma_thin", "Sigma_thick", "M50"}
QS = [1, 5, 16, 50, 84, 95, 99]


def _worker(rows, pot_cfg):
    from hydrabflow.simulators.stream_agama import _agama, _host_potential, _m200c_derived, _solar_frame

    ag = _agama()
    out = []
    for p in rows:
        pot = _host_potential(ag, p, pot_cfg)
        frame = _solar_frame(ag, pot, p)
        d = _m200c_derived(ag, p, pot_cfg, pot_host=pot, frame=frame)
        rec = {k[: -len("_derived")]: float(v) for k, v in d.items()}
        rec["Omega_sun"] = frame[2] / frame[0]  # (v_c(R0) + V_sun) / R0  [km/s/kpc]
        out.append(rec)
    return out


def mcmillan_reference(M200_from_halo=True):
    """McMillan (2017) best fit in the rc38 parameterization (agama's McMillan17.ini)."""
    from hydrabflow.simulators.stream_agama import Z_SUN_KPC, _agama
    from hydrabflow.simulators.stream_common import convert_concentration, vcirc_from_potential

    ag = _agama()
    ini = Path(ag.__file__).parent / "data" / "McMillan17.ini"
    pot = ag.Potential(file=str(ini))
    R0, V_sun = 8.21, 12.24
    S_thin, R_thin, z_thin = 8.95679e8, 2.49955, 0.3
    S_thick, R_thick, z_thick = 1.83444e8, 3.02134, 0.9
    rho = lambda S, z, R: S / (2 * z) * math.exp(-Z_SUN_KPC / z - R0 / R)  # noqa: E731
    bulge = ag.Potential(type="Spheroid", densityNorm=9.8351e10, axisRatioZ=0.5, gamma=0, beta=1.8,
                         scaleRadius=0.075, outerCutoffRadius=2.1)
    halo = ag.Potential(type="Spheroid", densityNorm=8.53702e6, axisRatioZ=1, gamma=1, beta=3,
                        scaleRadius=19.5725)
    H0 = 70.4 / 1000.0
    rho_crit = 3 * H0**2 / (8 * math.pi * ag.G)
    lo, hi = 50.0, 500.0  # r200 of the halo alone: M(<r) = 200 rho_crit (4 pi/3) r^3
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if halo.enclosedMass(mid) > 200 * rho_crit * 4 * math.pi / 3 * mid**3:
            lo = mid
        else:
            hi = mid
    r200 = 0.5 * (lo + hi)
    M200 = halo.enclosedMass(r200)
    c200 = r200 / 19.5725  # gamma = 1: r_-2 = r_h
    lo, hi = 1.0, 100.0  # invert convert_concentration(c_v', 94 -> 200) for c_v'
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if convert_concentration(mid, 94.0, 200.0) < c200:
            lo = mid
        else:
            hi = mid
    M_thin, M_thick = 2 * math.pi * S_thin * R_thin**2, 2 * math.pi * S_thick * R_thick**2
    M_b = bulge.totalMass()
    vc = float(vcirc_from_potential(pot, R0)[0])
    return {
        "gamma_TwoPowerTriaxial_halo": 1.0, "q_TwoPowerTriaxial_halo": 1.0,
        "log10_M200_TwoPowerTriaxial_halo": math.log10(M200),
        "ln_cvprime_TwoPowerTriaxial_halo": math.log(0.5 * (lo + hi)),
        "log10_Mstar": math.log10(M_thin + M_thick + M_b),
        "ln_R_d_thin": math.log(R_thin), "ln_R_d_thick": math.log(R_thick),
        "ln_f_thick": math.log(rho(S_thick, z_thick, R_thick) / rho(S_thin, z_thin, R_thin)),
        "R0_Sun": R0, "V_Sun": V_sun,
        "M_bulge": M_b, "M_disk": M_thin + M_thick, "M_thin": M_thin, "M_thick": M_thick,
        "Sigma_thin": S_thin, "Sigma_thick": S_thick, "R_d_thin": R_thin, "R_d_thick": R_thick,
        "f_thick": rho(S_thick, z_thick, R_thick) / rho(S_thin, z_thin, R_thin),
        "M50": float(pot.enclosedMass(50.0)), "vc_R0": vc, "Omega_sun": (vc + V_sun) / R0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from conftest import compose_cfg
    from joblib import Parallel, delayed

    from hydrabflow.registry import get_simulator

    sim = get_simulator(compose_cfg([f"simulator={SIM}"], fill=False).simulator)
    draws = sim.sample_prior(args.n, np.random.default_rng(args.seed))
    rows = [{k: float(v[i, 0]) for k, v in draws.items()} for i in range(args.n)]
    chunks = np.array_split(np.arange(args.n), args.n_workers * 4)
    res = Parallel(n_jobs=args.n_workers)(
        delayed(_worker)([rows[i] for i in c], sim._pot_cfg) for c in chunks if len(c)
    )
    recs = [r for part in res for r in part]
    der = {k: np.array([r[k] for r in recs]) for k in DERIVED}
    floored = np.array([r["disk_mass_floored"] for r in recs]) > 0.5
    samp = {k: draws[k][:, 0] for k in INFERRED + NUISANCE}
    ref = mcmillan_reference()

    Mstar = 10 ** samp["log10_Mstar"]
    below_bulge = Mstar < der["M_bulge"]
    lM = samp["log10_M200_TwoPowerTriaxial_halo"]
    lo_m, hi_m = (float(v) for v in sim._priors_global["log10_M200_TwoPowerTriaxial_halo"]["prior_parameters"])
    edges = np.linspace(lo_m, hi_m, int(round((hi_m - lo_m) / 0.1)) + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lM >= lo) & (lM < hi)
        vc = der["vc_R0"][m]
        bins.append({
            "log10_M200": [round(lo, 2), round(hi, 2)], "n": int(m.sum()),
            "frac_Mstar_below_Mbulge": float(below_bulge[m].mean()),
            "frac_floored": float(floored[m].mean()),
            "vc_R0_quantiles": dict(zip(map(str, QS), np.percentile(vc, QS).round(1).tolist())),
            "frac_vc_R0_below_180": float((vc < 180).mean()),
        })
    obs_r = np.asarray(sim.obs_r_kpc)
    i_obs = int(np.argmin(np.abs(obs_r - 8.2)))
    vc_obs = float(np.asarray(sim.obs_vc_kms)[i_obs])
    vc = der["vc_R0"]
    summary = {
        "n": args.n, "seed": args.seed, "simulator": SIM,
        "frac_Mstar_below_Mbulge": float(below_bulge.mean()),
        "frac_floored": float(floored.mean()),
        "per_log10_M200_bin": bins,
        "vc_R0_observed_table": {"R": float(obs_r[i_obs]), "vc": vc_obs},
        "frac_vc_R0_below_180": float((vc < 180).mean()),
        "frac_vc_R0_outside_180_280": float(((vc < 180) | (vc > 280)).mean()),
        "frac_vc_R0_within_10pct_of_obs": float((np.abs(vc / vc_obs - 1) < 0.1).mean()),
        "quantiles": {
            k: dict(zip(map(str, QS), np.percentile(v, QS).tolist()))
            for k, v in {**samp, **der}.items()
        },
        "mcmillan17_reference": ref,
        "any_nonfinite": {k: int((~np.isfinite(v)).sum()) for k, v in {**samp, **der}.items()},
        "any_negative_mass_or_sigma": int(sum((der[k] <= 0).sum() for k in
                                             ("M_disk", "M_thin", "M_thick", "Sigma_thin", "Sigma_thick"))),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(args.out, "prior_draws_potentials.npz"), floored=floored, **samp, **der)
    _plots(args.out, samp, der, floored, below_bulge, ref, bins, vc_obs)
    print(json.dumps({k: summary[k] for k in (
        "frac_Mstar_below_Mbulge", "frac_floored", "frac_vc_R0_below_180",
        "frac_vc_R0_outside_180_280", "frac_vc_R0_within_10pct_of_obs", "any_negative_mass_or_sigma")},
        indent=1))
    for b in bins:
        print(b["log10_M200"], "n=%d  M*<Mb %.3f  floored %.3f  vc_R0 p5/p50/p95 %s/%s/%s  <180: %.2f" % (
            b["n"], b["frac_Mstar_below_Mbulge"], b["frac_floored"], b["vc_R0_quantiles"]["5"],
            b["vc_R0_quantiles"]["50"], b["vc_R0_quantiles"]["95"], b["frac_vc_R0_below_180"]))
    print("McMillan17:", {k: (f"{v:.4g}") for k, v in ref.items()})


def _plots(out, samp, der, floored, below_bulge, ref, bins, vc_obs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def grid(data, keys, fname, title):
        n = len(keys)
        cols = 4
        rows = math.ceil(n / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        for ax, k in zip(axs.ravel(), keys):
            v = data[k]
            if k in LOG_AXES:
                v = np.log10(v)
                label = f"log10 {k}"
                r = ref.get(k)
                r = None if r is None else math.log10(r)
            else:
                label, r = k, ref.get(k)
            ax.hist(v, bins=80, color="#4c72b0", alpha=0.8)
            q = np.percentile(v, [5, 50, 95])
            for qq, ls in zip(q, (":", "-", ":")):
                ax.axvline(qq, color="k", ls=ls, lw=0.8)
            if r is not None:
                ax.axvline(r, color="#c44e52", lw=1.6, label="McMillan17")
                ax.legend(fontsize=7)
            ax.set_xlabel(label, fontsize=9)
            ax.set_yticks([])
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=120)
        plt.close(fig)

    grid(samp, INFERRED + NUISANCE, "sampled_parameters.png",
         "rc38 prior: sampled parameters (8 inferred + 4 nuisance); 5/50/95 %% lines")
    grid(der, DERIVED, "derived_quantities.png", "rc38 prior: derived quantities")

    lM = samp["log10_M200_TwoPowerTriaxial_halo"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ax = axs[0]
    ax.scatter(lM, der["vc_R0"], s=1, alpha=0.2, c=np.where(floored, "#c44e52", "#4c72b0"))
    mids = [np.mean(b["log10_M200"]) for b in bins]
    for q, ls in (("5", ":"), ("50", "-"), ("95", ":")):
        ax.plot(mids, [b["vc_R0_quantiles"][q] for b in bins], "k", ls=ls)
    ax.axhline(vc_obs, color="g", label=f"observed v_c(8.2 kpc) = {vc_obs:.0f}")
    ax.axhline(ref["vc_R0"], color="#c44e52", ls="--", label=f"McMillan17 {ref['vc_R0']:.0f}")
    ax.axhline(180, color="grey", lw=0.8)
    ax.set_xlabel("log10 M200 (halo only)")
    ax.set_ylabel("v_c(R0) [km/s]")
    ax.legend(fontsize=8)
    ax.set_title("red = M_disk floored")
    ax = axs[1]
    ax.plot(mids, [b["frac_floored"] for b in bins], "o-", label="on M_disk floor")
    ax.plot(mids, [b["frac_Mstar_below_Mbulge"] for b in bins], "s-", label="M* < M_bulge")
    ax.plot(mids, [b["frac_vc_R0_below_180"] for b in bins], "^-", label="v_c(R0) < 180")
    ax.set_xlabel("log10 M200")
    ax.set_ylabel("fraction")
    ax.legend()
    ax = axs[2]
    ax.scatter(lM, samp["log10_Mstar"], s=1, alpha=0.2)
    ax.axhline(math.log10(der["M_bulge"][0] + 1e9), color="#c44e52", lw=0.8, label="M_bulge + floor")
    ax.axhline(math.log10(der["M_bulge"][0]), color="k", lw=0.8, label="M_bulge")
    ax.plot(ref["log10_M200_TwoPowerTriaxial_halo"], ref["log10_Mstar"], "r*", ms=14, label="McMillan17")
    ax.set_xlabel("log10 M200")
    ax.set_ylabel("log10 M*")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "vc_R0_and_floor_vs_M200.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
