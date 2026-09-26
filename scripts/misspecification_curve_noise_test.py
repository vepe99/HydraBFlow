"""Does the summary-space MMD flag come from feeding the network a NOISE-FREE observed rotation
curve? The real-data chain has no `add_noise_to_vcirc`, while every training/reference curve
carries the observed per-radius sigma. Here the observed members are re-summarized through the
trained network with the training curve noise inserted before `log10_vcirc` (K seeds), and the
full / stream-slice / curve-slice tests are re-run against the same reference summaries.

Usage: python scripts/misspecification_curve_noise_test.py --real-run <eval_real dir> --d-stream 32
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-run", required=True)
    ap.add_argument("--d-stream", type=int, required=True)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--num-null", type=int, default=300)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from hydrabflow.pipeline import artifacts
    from hydrabflow.pipeline.adapter import fill_adapter_from_simulator, fill_stream_grid_from_simulator
    from hydrabflow.utils.paths import PREPROCESSING_STATE
    from hydrabflow.pipeline.compositional import apply_augmentations_once
    from hydrabflow.pipeline.evaluate_real import _prepare_real_members
    from hydrabflow.pipeline.misspecification import member_summaries, mmd_test, per_member_scores
    from hydrabflow.pipeline.workflow import build_workflow
    from hydrabflow.registry import build_pipeline

    cfg = OmegaConf.load(os.path.join(args.real_run, ".hydra", "config.yaml"))
    OmegaConf.set_struct(cfg, False)
    fill_adapter_from_simulator(cfg)
    fill_stream_grid_from_simulator(cfg)
    steps = list(cfg.augmentation.steps)
    names = [s if isinstance(s, str) else s.get("name") for s in steps]
    assert "add_noise_to_vcirc" not in names, "real chain already adds curve noise"
    steps.insert(names.index("log10_vcirc"), "add_noise_to_vcirc")
    cfg_noisy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_noisy.augmentation.steps = steps

    workflow = build_workflow(cfg)
    workflow.approximator = artifacts.load_approximator(cfg.model_dir)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(cfg.model_dir, PREPROCESSING_STATE))
    flat0, m = _prepare_real_members(cfg)
    flat0 = pipeline.transform(flat0)

    ref = np.load(os.path.join(str(cfg.eval.misspecification_reference), "summaries.npz"))
    R, rj = ref["summaries"], ref["j"].reshape(-1).astype(int)
    ds = args.d_stream
    Rc = R[::m, ds:]
    mu, inv = Rc.mean(0), np.linalg.pinv(np.cov(Rc, rowvar=False) + 1e-6 * np.eye(Rc.shape[1]))
    d_ref = np.sqrt(np.einsum("ij,jk,ik->i", Rc - mu, inv, Rc - mu))
    strm = {int(v): str(k) for k, v in cfg.simulator.params.target_streams.items()}

    def test(S, oj, tag):
        res = mmd_test(S, R, oj, rj, num_null=args.num_null, rng=np.random.default_rng(0))
        pm = per_member_scores(S, R, oj, rj)
        c = S[:1, ds:]
        d_obs = float(np.sqrt(((c - mu) @ inv @ (c - mu).T).item()))
        row = dict(mmd=res["mmd_observed"], p_strat=res.get("p_value_stratified"),
                   members={strm[k]: round(v["percentile"], 1) for k, v in pm.items()},
                   curve_mahal_pct=float((d_ref <= d_obs).mean() * 100))
        print(f"{tag:26s} mmd {row['mmd']:.3f} p_strat {row['p_strat']:.3f} members {row['members']} "
              f"curve-slice Mahalanobis pct {row['curve_mahal_pct']:.1f}")
        return row

    out = {}
    flat = apply_augmentations_once(dict(flat0), cfg, pipeline, int(cfg.seed))
    oj = np.asarray(flat["j"]).reshape(-1).astype(int)
    out["noise_free (as evaluate)"] = test(member_summaries(workflow.approximator, flat), oj, "noise-free (as evaluate)")
    rows = []
    for k in range(args.n_seeds):
        fl = apply_augmentations_once(dict(flat0), cfg_noisy, pipeline, int(cfg.seed) + 1 + k)
        rows.append(test(member_summaries(workflow.approximator, fl), oj, f"curve noise seed {k}"))
    out["curve_noise_seeds"] = rows
    agg = {"mmd_median": float(np.median([r["mmd"] for r in rows])),
           "p_strat_median": float(np.median([r["p_strat"] for r in rows])),
           "frac_seeds_p_lt_0.05": float(np.mean([r["p_strat"] < 0.05 for r in rows])),
           "curve_mahal_pct_median": float(np.median([r["curve_mahal_pct"] for r in rows]))}
    out["curve_noise_summary"] = agg
    print("SUMMARY over seeds:", agg)
    json.dump(out, open(os.path.join(args.real_run, "misspecification_curve_noise_test.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
