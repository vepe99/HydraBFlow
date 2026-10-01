#!/usr/bin/env python
"""phi2 vs phi1 for the three streams side by side (square panels) from a ppc_bspline_nn.py
bspline_features.npz: sim spline band, real B-spline (+/- its bootstrap sigma), real members.

  .venv/bin/python scripts/plot_bspline_phi2_row.py outputs/Bsline/.../ppc_stream_aug_core080
"""
import os, sys, json
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0"); os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import NAMES  # noqa: E402
from ppc_particle_coverage import real_clouds  # noqa: E402
from ppc_observation_space import stream_frames, to_obs  # noqa: E402

run = sys.argv[1]
args = open(os.path.join(run, "run.sh")).read().split()
opt = lambda k: args[args.index(k) + 1]
real = real_clouds(opt("--real"), opt("--simulator"), opt("--real-aug"), 1000, int(opt("--seed")))
R_of = stream_frames("streamfinder", opt("--real"))
d = np.load(os.path.join(run, "bspline_features.npz"))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (j, name) in zip(axes, NAMES.items()):
    g, S = d[f"{name}/grid"], d[f"{name}/sim_1"]
    S = S[np.all(np.isfinite(S), 1)]
    q = np.percentile(S, [5, 16, 50, 84, 95], axis=0)
    ax.fill_between(g, q[0], q[4], color="C0", alpha=0.2, label="sim 5-95 %")
    ax.fill_between(g, q[1], q[3], color="C0", alpha=0.35, label="sim 16-84 %")
    ax.plot(g, q[2], color="C0", lw=1.5, label="sim median")
    F = to_obs(R_of[j], real[j][0])
    ax.scatter(F[:, 0], F[:, 1], s=8, color="0.3", alpha=0.6, label="real members")
    r, s = d[f"{name}/real_1"], d[f"{name}/real_sigma_1"]
    ax.fill_between(g, r - s, r + s, color="C3", alpha=0.3, label="real ±1σ (bootstrap + errors)")
    ax.plot(g, r, color="C3", lw=2, label="real B-spline")
    l_, h_ = np.percentile(F[:, 1], [2, 98]); pad = 0.6 * (h_ - l_) + 1e-3
    ax.set_ylim(l_ - pad, h_ + pad)
    ax.set_title(f"{name} ({len(S)} sims)"); ax.set_xlabel(r"$\phi_1$ [deg]")
axes[0].set_ylabel(r"$\phi_2$ [deg]"); axes[0].legend(fontsize=7, loc="best")
fig.tight_layout()
out = os.path.join(run, "ppc_phi2_three_streams.png"); fig.savefig(out, dpi=150); print(out)
