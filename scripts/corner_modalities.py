"""Corner overlay of the three information sources behind the pooled compositional posterior.

    python scripts/corner_modalities.py <run_dir> [real] [reference.json]

<run_dir> holds `<real>/` (2nd arg, default eval_real) (compositional posterior.npz + preprocessing_state via train/),
`eval_real_only_sim_summary/` (single_stream_posterior.npz with the curve masked) and
`eval_real_only_vcirc_kms/` (single_stream_posterior.npz with the streams masked -> curve only;
identical for the 3 members, member 0 used). Chains are mapped to physical units through the
run's fitted preprocessing. Writes <run_dir>/<real>/corner_modalities.png. An optional reference.json
({"values": {param: value}}, e.g. McMillan 2017 in the same parameters) is drawn as truth lines and each
chain's percentile of it is printed.
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


def main(run_dir: str, real: str = "eval_real", reference: str | None = None) -> str:
    from hydrabflow.registry import build_pipeline, get_simulator
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(yaml.safe_load(open(os.path.join(run_dir, f"{real}/.hydra/config.yaml"))))
    # saved .hydra configs are pre-fill (inference_variables derive from the simulator at runtime)
    names = list(np.load(os.path.join(run_dir, f"{real}/posterior.npz")).files)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(run_dir, "train/preprocessing_state.npz"))
    stream_names = {int(v): str(k) for k, v in get_simulator(cfg.simulator).target_streams.items()}

    def phys(npz):
        d = pipeline.inverse_transform({k: np.asarray(npz[k]) for k in names})
        return d

    comp = phys(np.load(os.path.join(run_dir, f"{real}/posterior.npz")))
    streams = phys(np.load(os.path.join(run_dir, f"{real}_only_sim_summary/single_stream_posterior.npz")))
    curve = phys(np.load(os.path.join(run_dir, f"{real}_only_vcirc_kms/single_stream_posterior.npz")))
    j = np.load(cfg.data.real_data_path)["j"].reshape(-1).astype(int)

    stack = lambda d, i: np.column_stack([np.asarray(d[k])[i].reshape(-1) for k in names])  # noqa: E731
    chains = [(f"{stream_names.get(int(j[i]), i)} (curve masked)", stack(streams, i)) for i in range(len(j))]
    chains.append(("Rotation curve only", stack(curve, 0)))
    chains.append(("Compositional: 3 streams + curve", stack(comp, 0)))

    all_data = np.vstack([c for _, c in chains])
    lo, hi = np.nanpercentile(all_data, 0.5, axis=0), np.nanpercentile(all_data, 99.5, axis=0)
    pad = 0.05 * np.where(hi > lo, hi - lo, 1.0)
    ranges = list(zip(lo - pad, hi + pad))
    # the colors of evaluate_real's real_global_vs_streams_corner: RdYlBu_r over [Global, streams...]
    # (Global = the dark-blue end); the curve-only chain gets a teal-green that sits apart from that map
    cmap = plt.cm.RdYlBu_r(np.linspace(0, 1, len(j) + 1))
    colors = list(cmap[1:]) + ["#1b9e77", cmap[0]]
    fig = None
    for (label, data), c in zip(chains, colors):
        fig = corner.corner(
            data, fig=fig, labels=names, range=ranges, color=c, bins=40,
            plot_datapoints=False, plot_density=False, fill_contours=False, smooth=1.0,
            levels=(0.68, 0.95), hist_kwargs={"density": True, "lw": 2.0 if label.startswith("Comp") else 1.2},
            contour_kwargs={"linewidths": 2.0 if label.startswith("Comp") else 1.0},
        )
    handles = [Line2D([0], [0], color=c, label=lab) for (lab, _), c in zip(chains, colors)]
    if reference:
        import json
        ref = json.load(open(reference))["values"]
        truths = [ref.get(n) for n in names]
        corner.overplot_lines(fig, truths, color="0.25", ls="--", lw=1.2)
        corner.overplot_points(fig, [truths], marker="*", color="0.25", ms=9)
        handles.append(Line2D([0], [0], color="0.25", ls="--", marker="*", label="McMillan (2017)"))
        for label, data in chains:
            print(f"{label}: percentile of the reference", {n.split("_TwoPower")[0]: round(float((data[:, i] < truths[i]).mean() * 100), 1)
                                                            for i, n in enumerate(names) if truths[i] is not None})
    fig.legend(handles=handles,
               loc="upper right", fontsize=12, frameon=False)
    out = os.path.join(run_dir, real, "corner_modalities.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    for label, data in chains:
        q = np.percentile(data, [16, 50, 84], axis=0)
        print(label, {n.split("_")[0] if "_" in n and not n.startswith(("log10", "ln")) else n.split("_TwoPower")[0]:
                      f"{q[1, i]:.3g} [{q[0, i]:.3g},{q[2, i]:.3g}]" for i, n in enumerate(names)})
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:4]))
