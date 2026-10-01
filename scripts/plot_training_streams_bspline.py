#!/usr/bin/env python
"""N random training streams per stream with their B-spline tracks, against the real members.

Every row goes through the TRAINING observation model (``--aug`` chain up to ``mask_vlos``) and then the
model's own ``stream_bspline_grid`` (``--grid-aug``); the real members go through ``--real-aug`` and the
same grid. Published STREAMFINDER frames. Writes to --out:
  training_streams_bspline_<stream>.png   N panels of phi2 vs phi1: attended stars + that row's B-spline,
                                          real B-spline (black) and real members (grey) in every panel
  training_streams_bspline_overlay.png    3 streams x 5 observables: the N B-splines, real B-spline + members

  .venv/bin/python scripts/plot_training_streams_bspline.py --out outputs/Bsline/.../training_streams_bspline
"""
import argparse
import os
import sys

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_observation_space import read_rows, stream_frames, to_obs  # noqa: E402
from ppc_particle_coverage import compose_aug, real_clouds  # noqa: E402
from ppc_summary_statistics import NAMES, augment_sim  # noqa: E402

LABELS = [r"$\phi_2$ [deg]", r"$\varpi$ [mas]", r"$\mu_{\phi_1}$ [mas/yr]", r"$\mu_{\phi_2}$ [mas/yr]", r"$v_{\rm los}$ [km/s]"]
COLOR = dict(zip(NAMES.values(), plt.cm.RdYlBu_r(np.linspace(0, 1, 4))[1:]))
DARK = {"Pal5": "#2b7bba", "NGC3201": "#d9822b", "M68": "#7a0019"}   # darker twins for lines on the light stars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_ou24vc_hydrabflow/training_data_300000.npz")
    ap.add_argument("--simulator", default="stream_agama_spray_massloss_ibata_m200c_v4")
    ap.add_argument("--real", default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--aug", default="stream_global_streamfinder_bspline_core080")
    ap.add_argument("--real-aug", default="stream_real_global_streamfinder_bspline_core080")
    ap.add_argument("--grid-aug", default="stream_global_streamfinder_bspline_core080")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from omegaconf import OmegaConf
    from hydrabflow.registry import AUGMENTATIONS

    rng = np.random.default_rng(a.seed)
    with np.load(a.sim) as d:
        jf = np.asarray(d["j"]).reshape(-1).astype(int)
    rows = np.stack([np.sort(rng.choice(np.flatnonzero(jf == s), a.n, replace=False)) for s in range(3)], 1)   # (n, 3)
    raw = read_rows(a.sim, "sim_data_projected", rows.reshape(-1)).reshape(a.n, 3, -1, 6)
    jj = np.broadcast_to(np.arange(3), (a.n, 3))
    sim, attn, vmask = augment_sim(raw, jj, aug_preset=a.aug, simulator=a.simulator, seed=a.seed, upto="mask_vlos")

    gp = OmegaConf.to_container(compose_aug(a.simulator, a.grid_aug).params, resolve=True)
    grid_fn = AUGMENTATIONS.get("stream_bspline_grid")(OmegaConf.create(gp), np.random.default_rng(a.seed))
    key = gp.get("summary_key", "sim_summary")

    def grid(stars, att, vm, j):
        b = {"sim_data_projected": np.asarray(stars, np.float32), "attention_mask": np.asarray(att, np.float32)[:, None],
             "vlos_mask": np.asarray(vm, np.float32)[:, None], "j": np.full((len(stars), 1), j, np.float32)}
        o = np.asarray(grid_fn(b)[key])            # (n, G, 9): 5 obs, valid_track, valid_vlos, j, phi1 grid
        vals = {c: np.where((o[..., 6] if c == 4 else o[..., 5]) > 0, o[..., c], np.nan) for c in range(5)}
        return o[..., 8], vals

    real = real_clouds(a.real, a.simulator, a.real_aug, 1000, a.seed)
    R_of = stream_frames("streamfinder", a.real)
    fig_o, ax_o = plt.subplots(3, 5, figsize=(22, 11), squeeze=False)
    for s, (j, name) in enumerate(NAMES.items()):
        rst, rvm = real[j]
        g_r, v_r = grid(rst[None], np.ones((1, len(rst))), rvm[None], j)
        g_s, v_s = grid(sim[:, s], attn[:, s], vmask[:, s], j)
        Fr = to_obs(R_of[j], rst)
        gv = np.linspace(Fr[rvm, 0].min(), Fr[rvm, 0].max(), g_r.shape[1])   # v_los grid = real measured range
        xlim = (Fr[:, 0].min() - 3, Fr[:, 0].max() + 3)

        # --- per-stream grid of phi2 panels
        nc = int(np.ceil(np.sqrt(a.n))); nr = int(np.ceil(a.n / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(2.2 * nc, 1.8 * nr), sharex=True, sharey=True, squeeze=False)
        lo, hi = np.percentile(Fr[:, 1], [1, 99]); pad = 0.8 * (hi - lo)
        for k, ax in enumerate(axes.flat):
            if k >= a.n:
                ax.set_visible(False); continue
            F = to_obs(R_of[j], sim[k, s][attn[k, s] > 0])
            ax.scatter(Fr[:, 0], Fr[:, 1], s=1.5, color="0.6", lw=0)
            ax.scatter(F[:, 0], F[:, 1], s=1.5, color=COLOR[name], lw=0)
            ax.plot(g_r[0], v_r[0][0], color="black", lw=1.2)
            ax.plot(g_s[k], v_s[0][k], color=DARK[name], lw=1.2)
            ax.set(xlim=xlim, ylim=(lo - pad, hi + pad))
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=6)
        for ax in axes[-1]:
            ax.set_xlabel(r"$\phi_1$ [deg]", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel(LABELS[0], fontsize=8)
        fig.subplots_adjust(wspace=0.05, hspace=0.08)
        p = os.path.join(a.out, f"training_streams_bspline_{name}.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(p)

        # --- overlay: all N splines per observable
        for c in range(5):
            ax = ax_o[s, c]
            x_s, x_r = (np.broadcast_to(gv, g_s.shape), gv) if c == 4 else (g_s, g_r[0])
            ok_r = np.isfinite(Fr[:, c + 1]) & (rvm if c == 4 else True)
            ax.scatter(Fr[ok_r, 0], Fr[ok_r, c + 1], s=6, color="0.55", lw=0, zorder=1, label="Gaia members")
            for k in range(a.n):
                ax.plot(x_s[k], v_s[c][k], color=COLOR[name], lw=0.7, alpha=0.5, zorder=2)
            ax.plot(x_r, v_r[c][0], color="black", lw=2.2, zorder=3, label="Gaia B-spline")
            vv = np.concatenate([v_s[c][np.isfinite(v_s[c])], v_r[c][0][np.isfinite(v_r[c][0])]])
            lo_, hi_ = np.percentile(vv, [1, 99]); pd = 0.25 * (hi_ - lo_)
            ax.set(xlim=xlim, ylim=(lo_ - pd, hi_ + pd), ylabel=LABELS[c])
            ax.spines[["top", "right"]].set_visible(False)
            if s == 2:
                ax.set_xlabel(r"$\phi_1$ [deg]")
            if c == 0:
                ax.text(0.03, 0.95, name, transform=ax.transAxes, fontweight="bold", va="top")
    ax_o[0, 4].plot([], [], color=COLOR["Pal5"], lw=0.7, label=f"{a.n} training streams")
    ax_o[0, 4].legend(frameon=False, fontsize=8, loc="upper right")
    fig_o.tight_layout()
    p = os.path.join(a.out, "training_streams_bspline_overlay.png"); fig_o.savefig(p, dpi=150, bbox_inches="tight")
    print(p)


if __name__ == "__main__":
    main()
