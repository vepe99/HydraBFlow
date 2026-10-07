"""Posteriors of derived Galaxy quantities for an m200_c global run, per information source.

For each draw: the halo (scaleRadius r_h, densityNorm rho) AGAMA actually receives, the enclosed mass
M(<R_ENC) of the FULL potential and of the halo alone, and the local dark-matter density
rho_DM(R0, 0, 0) at that draw's R0_Sun (a marginalized nuisance, drawn from its prior). Potentials are
built exactly as the simulator does (scripts/ppc_rotation_curve_ibata.py helpers).

Sources: <dir>/eval_real (posterior.npz = compositional), <dir>/eval_real_only_sim_summary
(single_stream_posterior.npz = each stream alone, curve masked), <dir>/eval_real_only_vcirc_kms
(curve only, member 0). Writes <dir>/derived_halo_posterior.{json,png}.

    JAX_PLATFORMS=cpu .venv/bin/python scripts/derived_halo_posterior.py <dir> [--n 400] [--ref mcmillan17_reference.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ppc_rotation_curve_ibata import _identity_constants, _load_posterior, _log10_keys  # noqa: E402

R_ENC = 20.0
GEV_PER_MSUN_PC3 = 37.97  # 1 Msun/pc^3 in GeV/cm^3
QTY = [("r_h", "r_h [kpc]"), ("rho", r"$\rho_0$ (AGAMA densityNorm) [M$_\odot$/kpc$^3$]"),
       ("M20_total", r"M(<20 kpc) total [10$^{11}$ M$_\odot$]"), ("M20_halo", r"M(<20 kpc) halo [10$^{11}$ M$_\odot$]"),
       ("rho_dm_sun", r"$\rho_{\rm DM}(R_0)$ [GeV/cm$^3$]")]

_G = {}


def _init(pot_cfg):
    from hydrabflow.simulators.stream_agama import _agama, _resolve_pot_cfg
    _G["agama"] = _agama(); _G["agama"].setNumThreads(1)
    _G["pot_cfg"], _G["rcfg"] = pot_cfg, _resolve_pot_cfg(pot_cfg)


def _derive(row):
    from hydrabflow.simulators.stream_agama import _halo_params_m200c, _host_potential
    agama = _G["agama"]
    try:
        hp = _halo_params_m200c(agama, row, _G["rcfg"])
        halo = agama.Potential(hp)
        full = _host_potential(agama, row, _G["pot_cfg"])
        R0 = float(row.get("R0_Sun", 8.178))
        rho_sun = float(halo.density([R0, 0.0, 0.0])) * 1e-9 * GEV_PER_MSUN_PC3
        return [float(hp["scaleRadius"]), float(hp["densityNorm"]), full.enclosedMass(R_ENC) / 1e11,
                halo.enclosedMass(R_ENC) / 1e11, rho_sun]
    except Exception:  # e.g. a leaked gamma >= 2 makes r_h non-positive
        return [np.nan] * len(QTY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--n", type=int, default=400); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ref", default=None, help="reference.json in the same parameters (e.g. McMillan 2017)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    from multiprocessing import Pool
    from omegaconf import OmegaConf
    from hydrabflow.registry import get_simulator

    cfg = OmegaConf.load(os.path.join(a.dir, "eval_real", ".hydra", "config.yaml"))
    params = OmegaConf.to_container(cfg.simulator.params, resolve=True)
    sim = get_simulator(cfg.simulator)
    pot_cfg = sim._pot_cfg
    consts = _identity_constants(params["priors_global"])
    nuis = {k: params["priors_global"][k] for k in params.get("marginalize") or []}
    rng = np.random.default_rng(a.seed)
    names = {int(v): str(k) for k, v in params["target_streams"].items()}
    lk = _log10_keys(OmegaConf.to_container(cfg, resolve=True))

    def rows(post, g):
        keys = list(post)
        idx = rng.choice(post[keys[0]].shape[1], size=min(a.n, post[keys[0]].shape[1]), replace=False)
        out = []
        for i in idx:
            r = {**consts, **{k: float(post[k][g, i]) for k in keys}}
            for k, s in nuis.items():
                p0, p1 = (float(v) for v in s["prior_parameters"])
                r[k] = {"normal": rng.normal, "uniform": rng.uniform}[str(s["type"])](p0, p1)
            out.append(r)
        return out

    streams = _load_posterior(os.path.join(a.dir, "eval_real_only_sim_summary", "single_stream_posterior.npz"), lk)
    curve = _load_posterior(os.path.join(a.dir, "eval_real_only_vcirc_kms", "single_stream_posterior.npz"), lk)
    comp = _load_posterior(os.path.join(a.dir, "eval_real", "posterior.npz"), lk)
    sources = [(f"{names[g]} (curve masked)", rows(streams, g)) for g in range(streams[next(iter(streams))].shape[0])]
    sources += [("Rotation curve only", rows(curve, 0)), ("Compositional: 3 streams + curve", rows(comp, 0))]

    if a.ref:  # reference point (McMillan 2017) through the same conversion
        ref_row = {**consts, **json.load(open(a.ref))["values"], "R0_Sun": 8.21}
        sources.append(("McMillan (2017)", [ref_row]))

    res = {}
    with Pool(a.workers, initializer=_init, initargs=(pot_cfg,)) as pool:
        for label, rs in sources:
            res[label] = np.array(pool.map(_derive, rs))
            print(f"{label}: {np.isfinite(res[label][:, 0]).sum()}/{len(rs)} draws")

    summary = {}
    print(f"\n{'source':34s} " + " ".join(f"{q:>24s}" for q, _ in QTY))
    for label, v in res.items():
        if len(v) == 1:
            summary[label] = {q: float(v[0, i]) for i, (q, _) in enumerate(QTY)}
            print(f"{label:34s} " + " ".join(f"{v[0, i]:24.4g}" for i in range(len(QTY))))
            continue
        q = np.nanpercentile(v, [16, 50, 84], axis=0)
        summary[label] = {n: {"p16": q[0, i], "median": q[1, i], "p84": q[2, i]} for i, (n, _) in enumerate(QTY)}
        print(f"{label:34s} " + " ".join(f"{q[1, i]:9.4g} [{q[0, i]:.3g},{q[2, i]:.3g}]".rjust(24) for i in range(len(QTY))))
    json.dump(summary, open(os.path.join(a.dir, "derived_halo_posterior.json"), "w"), indent=1, default=float)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.cm.RdYlBu_r(np.linspace(0, 1, 4))   # same colors as corner_modalities.py
    colors = {l: c for l, c in zip([s for s, _ in sources[:3]], cmap[1:])}
    colors["Rotation curve only"], colors["Compositional: 3 streams + curve"] = "#1b9e77", cmap[0]
    fig, axes = plt.subplots(1, len(QTY), figsize=(4.2 * len(QTY), 3.6))
    for i, (q, lab) in enumerate(QTY):
        ax = axes[i]
        allv = np.concatenate([v[:, i] for l, v in res.items() if len(v) > 1])
        lo, hi = np.nanpercentile(allv, [0.5, 99.5])
        bins = np.geomspace(lo, hi, 40) if q in ("r_h", "rho") else np.linspace(lo, hi, 40)
        for l, v in res.items():
            if len(v) == 1:
                ax.axvline(v[0, i], color="0.25", ls="--", lw=1.2, label=l)
            else:
                ax.hist(v[:, i], bins=bins, histtype="step", density=True, color=colors[l],
                        lw=2.2 if l.startswith("Comp") else 1.3, label=l)
        if q in ("r_h", "rho"):
            ax.set_xscale("log")
        ax.set_xlabel(lab); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    out = os.path.join(a.dir, "derived_halo_posterior.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
