"""Nearest training rows to the real Gaia streams in the trained model's LEARNED summary space.

Answers "why does the network read a prolate halo off this stream?": N flat training rows go through
the exact training chain (preprocessing transform + augmentation, as mmd_stream_vs_training.py), the
real members through theirs, both through the trained summary network; then, per stream, the k rows
nearest to the real stream in the STREAM slice of the fused summary (the first `--stream-dim`
outputs, i.e. the sim_summary backbone) are reported with their parameters against the prior, plus the
curve-slice neighbours of the real rotation curve. Output: `nn_learned_summary.{json,png}` + a table.

    HYDRABFLOW_NUM_GPUS=0 .venv/bin/python scripts/nn_learned_summary.py --run outputs/v4_2modal_rnbody_mlp \
        --train data/data_jarvis/data_agama_rnbody_ibata_m200c_v4_hydrabflow/training_data_100000.npz
"""
from __future__ import annotations

import argparse, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from nearest_training_streams import GLOBALS, LOCALS  # noqa: E402

NAMES = {0: "Pal5", 1: "NGC3201", 2: "M68"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="runs dir holding train/, eval_sim_333/, eval_real/")
    ap.add_argument("--train", required=True, help="flat training .npz")
    ap.add_argument("--n-ref", type=int, default=20000); ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--stream-dim", type=int, default=32, help="width of the sim_summary backbone (first slice of the fused summary)")
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = a.out or os.path.join(a.run, "eval_real"); os.makedirs(out, exist_ok=True)

    from omegaconf import OmegaConf
    from hydrabflow.pipeline.adapter import fill_adapter_from_simulator, fill_stream_grid_from_simulator
    from hydrabflow.pipeline.artifacts import load_approximator
    from hydrabflow.pipeline.compositional import apply_augmentations_once
    from hydrabflow.pipeline.evaluate_real import _prepare_real_members
    from hydrabflow.pipeline.misspecification import member_summaries
    from hydrabflow.registry import build_pipeline
    from hydrabflow.utils.paths import PREPROCESSING_STATE

    def load_cfg(run):
        c = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml")); OmegaConf.set_struct(c, False)
        fill_adapter_from_simulator(c); fill_stream_grid_from_simulator(c); return c

    import bayesflow  # noqa: F401  registers bayesflow's keras classes for load_model
    import hydrabflow.networks.fusion, hydrabflow.networks.grouped_diffusion, hydrabflow.networks.masked_time_series_transformer, hydrabflow.networks.factory  # noqa: F401,E401
    from hydrabflow.pipeline._bf_patches import apply_bayesflow_patches; apply_bayesflow_patches()
    md = os.path.join(a.run, "train"); cs = load_cfg(os.path.join(a.run, "eval_sim_333")); cr = load_cfg(os.path.join(a.run, "eval_real"))
    from hydrabflow.pipeline.workflow import build_workflow
    build_workflow(cs)                 # builds the nets once -> registers the lazily-defined serializable classes (as evaluate does)
    approx = load_approximator(md)

    # --- training rows through the training chain ---
    raw = np.load(a.train, allow_pickle=True); n_tot = np.asarray(raw["j"]).shape[0]
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(n_tot, size=min(a.n_ref, n_tot), replace=False))
    ref = {k: np.asarray(raw[k])[idx] for k in raw.files}; ref["j"] = np.asarray(raw["j"], float).reshape(n_tot, 1)[idx]
    params_all = {k: np.asarray(raw[k]).reshape(n_tot, -1)[:, 0] for k in GLOBALS + LOCALS if k in raw.files}
    keep = np.isfinite(ref["sim_data_projected"].reshape(len(idx), -1)).any(axis=1); idx = idx[keep]
    ref = {k: v[keep] for k, v in ref.items()}
    pipe = build_pipeline(cs.preprocessing); pipe.load(os.path.join(md, PREPROCESSING_STATE))
    ref = apply_augmentations_once(pipe.transform(ref), cs, pipe, int(cs.seed))
    print(f"summarizing {len(idx)} training rows ...", flush=True)
    S = np.concatenate([member_summaries(approx, {k: v[i:i + 2000] for k, v in ref.items()}) for i in range(0, len(idx), 2000)])
    jr = np.asarray(ref["j"]).reshape(-1).astype(int)

    # --- real members through theirs ---
    rpipe = build_pipeline(cr.preprocessing); rpipe.load(os.path.join(md, PREPROCESSING_STATE))
    real, _ = _prepare_real_members(cr); real = apply_augmentations_once(rpipe.transform(real), cr, rpipe, int(cr.seed))
    So = member_summaries(approx, real); jo = np.asarray(real["j"]).reshape(-1).astype(int)
    print(f"summary width {S.shape[1]}: stream slice [:{a.stream_dim}], curve slice [{a.stream_dim}:]")

    slices = {"stream": slice(0, a.stream_dim), "curve": slice(a.stream_dim, S.shape[1]), "fused": slice(0, S.shape[1])}
    report = {"n_ref": int(len(idx)), "k": a.k}
    fig, axes = plt.subplots(3, len(slices), figsize=(5 * len(slices), 10), squeeze=False)
    for r, j in enumerate(jo):
        name = NAMES.get(int(j), str(j)); same = jr == j; report[name] = {}
        for c, (sl, s) in enumerate(slices.items()):
            X = S[same][:, s]; x = So[r][s]
            sd = X.std(axis=0) + 1e-12; d = np.sqrt((((X - x) / sd) ** 2).sum(axis=1))
            nn = np.argsort(d)[: a.k]; rows = idx[same][nn]
            # typicality: real->nearest vs sim->nearest (leave-one-out) in the same space
            Z = X / sd; sub = rng.choice(len(Z), size=min(2000, len(Z)), replace=False)
            dd = np.sqrt(((Z[sub][:, None] - Z[None]) ** 2).sum(-1)); dd[np.arange(len(sub)), sub] = np.inf
            typ = float((dd.min(1) < d[nn[0]]).mean() * 100)
            tab = {}
            for p, v in params_all.items():
                kn = v[rows]; pr = v[idx[same]]
                if not np.isfinite(pr).any(): continue
                q = lambda z: np.nanpercentile(z, [16, 50, 84]).tolist()
                psd = np.nanstd(pr); tab[p] = {"prior": q(pr), "nn": q(kn), "shift_sd": float((np.nanmedian(kn) - np.nanmedian(pr)) / psd) if psd > 0 else 0.0}
            report[name][sl] = {"typicality_pct": typ, "nearest_dist": float(d[nn[0]]), "params": tab, "rows": rows.tolist()}
            print(f"\n== {name} / {sl} slice: real at the {typ:.0f}th pct of nearest-neighbour distance")
            print(f"  {'param':34s} {'prior med [16,84]':>28s} {'k-NN med [16,84]':>28s}  shift/sd")
            for p, t in tab.items():
                f = lambda v: f"{v[1]:9.4g} [{v[0]:9.3g},{v[2]:9.3g}]"
                print(f"  {p:34s} {f(t['prior']):>28s} {f(t['nn']):>28s}  {t['shift_sd']:+.2f}")
            ax = axes[r, c]; qk = "q_TwoPowerTriaxial_halo"
            ax.hist(params_all[qk][idx[same]], bins=30, density=True, alpha=0.3, label="prior (training)")
            ax.hist(params_all[qk][rows], bins=15, density=True, alpha=0.6, label=f"{a.k} nearest")
            ax.set_title(f"{name}: {sl} slice (typ. {typ:.0f} pct)"); ax.set_xlabel("q_halo")
            if r == 0 and c == 0: ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "nn_learned_summary.png"), dpi=130)
    json.dump(report, open(os.path.join(out, "nn_learned_summary.json"), "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
