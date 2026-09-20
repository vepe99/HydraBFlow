"""Corner overlay of the three information sources behind the pooled compositional posterior.

    python scripts/corner_modalities.py <run_dir>

<run_dir> holds `eval_real/` (compositional posterior.npz + preprocessing_state via train/),
`eval_real_only_sim_summary/` (single_stream_posterior.npz with the curve masked) and
`eval_real_only_vcirc_kms/` (single_stream_posterior.npz with the streams masked -> curve only;
identical for the 3 members, member 0 used). Chains are mapped to physical units through the
run's fitted preprocessing. Writes <run_dir>/corner_modalities.png.
"""

from __future__ import annotations

import os
import sys

import corner
import matplotlib
import numpy as np
import yaml
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(run_dir: str) -> str:
    from hydrabflow.registry import build_pipeline, get_simulator
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(yaml.safe_load(open(os.path.join(run_dir, "eval_real/.hydra/config.yaml"))))
    # saved .hydra configs are pre-fill (inference_variables derive from the simulator at runtime)
    names = list(np.load(os.path.join(run_dir, "eval_real/posterior.npz")).files)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(run_dir, "train/preprocessing_state.npz"))
    stream_names = {int(v): str(k) for k, v in get_simulator(cfg.simulator).target_streams.items()}

    def phys(npz):
        d = pipeline.inverse_transform({k: np.asarray(npz[k]) for k in names})
        return d

    comp = phys(np.load(os.path.join(run_dir, "eval_real/posterior.npz")))
    streams = phys(np.load(os.path.join(run_dir, "eval_real_only_sim_summary/single_stream_posterior.npz")))
    curve = phys(np.load(os.path.join(run_dir, "eval_real_only_vcirc_kms/single_stream_posterior.npz")))
    j = np.load(cfg.data.real_data_path)["j"].reshape(-1).astype(int)

    stack = lambda d, i: np.column_stack([np.asarray(d[k])[i].reshape(-1) for k in names])  # noqa: E731
    chains = [(f"{stream_names.get(int(j[i]), i)} (curve masked)", stack(streams, i)) for i in range(len(j))]
    chains.append(("Rotation curve only", stack(curve, 0)))
    chains.append(("Compositional: 3 streams + curve", stack(comp, 0)))

    all_data = np.vstack([c for _, c in chains])
    lo, hi = np.nanpercentile(all_data, 0.5, axis=0), np.nanpercentile(all_data, 99.5, axis=0)
    pad = 0.05 * np.where(hi > lo, hi - lo, 1.0)
    ranges = list(zip(lo - pad, hi + pad))
    colors = list(plt.cm.RdYlBu_r(np.linspace(0.75, 1.0, len(j)))) + ["tab:green", "black"]
    fig = None
    for (label, data), c in zip(chains, colors):
        fig = corner.corner(
            data, fig=fig, labels=names, range=ranges, color=c, bins=40,
            plot_datapoints=False, plot_density=False, fill_contours=False, smooth=1.0,
            levels=(0.68, 0.95), hist_kwargs={"density": True, "lw": 2.0 if label.startswith("Comp") else 1.2},
            contour_kwargs={"linewidths": 2.0 if label.startswith("Comp") else 1.0},
        )
    fig.legend(handles=[Line2D([0], [0], color=c, label=lab) for (lab, _), c in zip(chains, colors)],
               loc="upper right", fontsize=12, frameon=False)
    out = os.path.join(run_dir, "corner_modalities.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    for label, data in chains:
        q = np.percentile(data, [16, 50, 84], axis=0)
        print(label, {n.split("_")[0] if "_" in n and not n.startswith(("log10", "ln")) else n.split("_TwoPower")[0]:
                      f"{q[1, i]:.3g} [{q[0, i]:.3g},{q[2, i]:.3g}]" for i, n in enumerate(names)})
    return out


if __name__ == "__main__":
    print(main(sys.argv[1]))
