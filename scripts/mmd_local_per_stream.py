"""Summary-space misspecification test of a LOCAL-level model, one stream at a time.

The local evaluate stages save no summaries, so this runs only the summary network (no posterior
sampling): the grouped test set through the local model's own sim-eval chain (flatten + preprocessing
+ one augmentation draw, as ``_evaluate_local``), the real members through the real-eval chain (as
``_evaluate_real_compositional``). The summaries do not depend on the conditioning globals, so any
real-eval run of the model serves. Per stream and per slice of the fused summary (stream backbone /
curve backbone / fused), the observed member is compared with that stream's simulated members:

* MMD(observed, reference) with a LEAVE-ONE-OUT null: one held-out reference member vs the rest
  (the stock test draws its null from inside the reference, which is anti-conservative);
* Mahalanobis distance percentile, also leave-one-out (each reference member scored against the mean
  and covariance of the others).

    JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0 .venv/bin/python scripts/mmd_local_per_stream.py \
        --model-dir <local trial> --real-run <pair>/eval_real --out <dir>
"""
from __future__ import annotations

import argparse, json, os
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
import numpy as np


def loo_mahalanobis(ref, x, ridge=1e-3):
    """Percentile of x's distance among the reference members' leave-one-out distances."""
    def dist(pool, y):
        mu = pool.mean(0); cov = np.cov(pool, rowvar=False)
        cov += ridge * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        d = y - mu; return float(np.sqrt(d @ np.linalg.pinv(cov) @ d))
    d_obs = dist(ref, x)
    d_ref = np.array([dist(np.delete(ref, i, 0), ref[i]) for i in range(len(ref))])
    return d_obs, float(np.median(d_ref)), float((d_ref <= d_obs).mean() * 100), d_ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="local trial dir (approximator + eval_sim_333/.hydra)")
    ap.add_argument("--real-run", required=True, help="a real-data local eval run of this model (its .hydra config)")
    ap.add_argument("--sim-run", default=None, help="sim-eval run whose config builds the reference (default <model-dir>/eval_sim_333)")
    ap.add_argument("--num-null", type=int, default=300); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    from omegaconf import OmegaConf
    from hydrabflow.pipeline.adapter import fill_adapter_from_simulator, fill_stream_grid_from_simulator
    from hydrabflow.pipeline.artifacts import load_approximator
    from hydrabflow.pipeline.compositional import apply_augmentations_once, flatten_members
    from hydrabflow.pipeline.evaluate import _load_test_data, stream_names
    from hydrabflow.pipeline.evaluate_real import _prepare_real_members
    from hydrabflow.pipeline.misspecification import _bf_mmd, member_summaries
    from hydrabflow.pipeline.workflow import build_workflow
    from hydrabflow.pipeline._bf_patches import apply_bayesflow_patches
    from hydrabflow.registry import build_pipeline
    from hydrabflow.utils.paths import PREPROCESSING_STATE

    def load_cfg(run):
        c = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml")); OmegaConf.set_struct(c, False)
        fill_adapter_from_simulator(c); fill_stream_grid_from_simulator(c); return c

    apply_bayesflow_patches()
    cs, cr = load_cfg(a.sim_run or os.path.join(a.model_dir, "eval_sim_333")), load_cfg(a.real_run)
    build_workflow(cs)  # registers the lazily-defined serializable classes before loading (as evaluate does)
    approx = load_approximator(a.model_dir)

    test, pipe = _load_test_data(cs, a.model_dir)
    flat = apply_augmentations_once(flatten_members(test, np.asarray(test["j"]).shape[1]), cs, pipe, int(cs.seed))
    S = np.concatenate([member_summaries(approx, {k: np.asarray(v)[i:i + 256] for k, v in flat.items()})
                        for i in range(0, len(np.asarray(flat["j"])), 256)])
    jr = np.asarray(flat["j"]).reshape(-1).astype(int)

    rpipe = build_pipeline(cr.preprocessing); rpipe.load(os.path.join(a.model_dir, PREPROCESSING_STATE))
    real, _ = _prepare_real_members(cr)
    real = apply_augmentations_once(rpipe.transform(real), cr, rpipe, int(cr.seed))
    So = member_summaries(approx, real); jo = np.asarray(real["j"]).reshape(-1).astype(int)

    # slice widths follow adapter.summary_variables order (fusion head null -> concatenated backbones)
    bb = cs.model.summary_network.params.backbones; edges, lo = {}, 0
    for key in cs.adapter.summary_variables:
        w = int(bb[key].summary_dim); edges[key] = slice(lo, lo + w); lo += w
    assert lo == S.shape[1], f"slice widths {lo} != summary width {S.shape[1]}"
    edges["fused"] = slice(0, S.shape[1])

    names = stream_names(cs, jo); rng = np.random.default_rng(a.seed); report = {}; dists = {}
    print(f"reference: {len(jr)} simulated members; summary width {S.shape[1]}")
    print(f"{'stream':8s} {'slice':20s} {'MMD':>7s} {'null95':>7s} {'p(LOO)':>7s} {'Mahal':>6s} {'ref med':>7s} {'pct(LOO)':>8s}")
    for r, j in enumerate(jo):
        name = names[int(j)]; report[name] = {}
        for sl, s in edges.items():
            ref = S[jr == j][:, s]; x = So[r:r + 1, s]
            mmd = _bf_mmd(x, ref)
            null = np.array([_bf_mmd(ref[i:i + 1], np.delete(ref, i, 0))
                             for i in rng.integers(0, len(ref), a.num_null)])
            d, dmed, pct, d_ref = loo_mahalanobis(ref, x[0]); dists[name, sl] = (null, d_ref)
            report[name][sl] = {"mmd": mmd, "null_95": float(np.quantile(null, 0.95)), "p_value": float((null >= mmd).mean()),
                                "mahalanobis": d, "reference_median": dmed, "percentile": pct, "n_reference": int(len(ref))}
            t = report[name][sl]
            print(f"{name:8s} {sl:20s} {mmd:7.3f} {t['null_95']:7.3f} {t['p_value']:7.3f} {d:6.2f} {dmed:7.2f} {pct:8.1f}")
    json.dump(report, open(os.path.join(a.out, "mmd_local_per_stream.json"), "w"), indent=1)
    print("wrote", os.path.join(a.out, "mmd_local_per_stream.json"))
    plot(report, dists, os.path.join(a.out, "mmd_local_per_stream.png"))


def plot(report, dists, path):
    """Rows = streams, columns = (MMD, Mahalanobis) x slice: LOO reference histogram + observed line."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(report); slices = list(report[names[0]])
    fig, ax = plt.subplots(len(names), 2 * len(slices), figsize=(3.2 * 2 * len(slices), 2.6 * len(names)), squeeze=False)
    for r, n in enumerate(names):
        for c, sl in enumerate(slices):
            t = report[n][sl]; null, d_ref = dists[n, sl]
            for k, (vals, obs, lab, txt) in enumerate([
                    (null, t["mmd"], "MMD (LOO null)", f"p = {t['p_value']:.3f}"),
                    (d_ref, t["mahalanobis"], "Mahalanobis (LOO ref)", f"pct = {t['percentile']:.1f}")]):
                a = ax[r, 2 * c + k]
                a.hist(vals, bins=30, color="0.7", density=True)
                a.axvline(obs, color="C3", lw=2)
                a.set_title(f"{n} | {sl}\n{txt}", fontsize=9); a.set_yticks([])
                if r == len(names) - 1: a.set_xlabel(lab, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig); print("wrote", path)


if __name__ == "__main__":
    main()
