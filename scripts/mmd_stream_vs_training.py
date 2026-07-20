"""Stream-channel (sim_summary) misspecification MMD vs a large training-set reference.

The potential channels were handled noise-free per-potential (scripts/mmd_potential_vs_training.py).
The STREAM channel is different: sim_summary is derived from the star cloud through the Gaia
observation model (windows / magnitudes / errors / vlos masking) + the phi1-binning augmentation, so
it must be recomputed by running that augmentation chain -- and it is the one channel where MMD has
real power, because the observed group holds 3 genuinely independent members (one per stream) and
the reference holds thousands per stream.

Reference = N (default 10^4) rows of the FLAT training set, pushed through the SAME augmentation
chain the network trained on (apply_augmentations_once with the sim config) -> sim_summary + j.
Observed = the real Gaia group through the real preprocessing + augmentation (as evaluate_real does).
Then the Schmitt+21 MMD test (bayesflow MMD, j-stratified bootstrap null) + per-stream Mahalanobis
percentiles, and the mmd_hypothesis_test plot.

Imports hydrabflow (needs the augmentation chain), forced onto CPU. Usage:
    uv run python scripts/mmd_stream_vs_training.py \
        --sim-run  outputs/ibata_onedisk_grid_m200c/default/eval_sim_333 \
        --real-run outputs/ibata_onedisk_grid_m200c/default/eval_real \
        --train    data/data_jarvis/data_agama_ibata_onedisk_beta3_m200c_hydrabflow/training_data_100000.npz \
        --n-ref 10000
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-run", required=True, help="evaluate composition=global run dir (for the sim config)")
    ap.add_argument("--real-run", required=True, help="evaluate_real run dir (observed group)")
    ap.add_argument("--train", required=True, help="flat training .npz (reference source)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--n-ref", type=int, default=10000)
    ap.add_argument("--num-null", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from hydrabflow.pipeline.adapter import (
        fill_adapter_from_simulator, fill_stream_grid_from_simulator,
    )
    from hydrabflow.pipeline.compositional import apply_augmentations_once
    from hydrabflow.pipeline.evaluate_real import _prepare_real_members
    from hydrabflow.pipeline.misspecification import mmd_test, per_member_scores
    from hydrabflow.preprocessing.registry import build_pipeline
    from hydrabflow.utils.paths import PREPROCESSING_STATE

    def load_cfg(run):
        c = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml"))
        OmegaConf.set_struct(c, False)
        fill_adapter_from_simulator(c)
        fill_stream_grid_from_simulator(c)
        return c

    cs = load_cfg(args.sim_run)
    cr = load_cfg(args.real_run)
    md = args.model_dir or cs.get("model_dir") or os.path.join(os.path.dirname(args.sim_run), "train")

    sim_pipe = build_pipeline(cs.preprocessing)
    sim_pipe.load(os.path.join(md, PREPROCESSING_STATE))

    # --- reference: N flat training rows through the training-time augmentation chain ---
    raw = np.load(args.train, allow_pickle=True)
    n_tot = np.asarray(raw["j"]).shape[0]
    # drop rows whose star cloud is all-NaN (the handful of failed sims)
    sd = np.asarray(raw["sim_data_projected"], float)
    finite = np.isfinite(sd.reshape(n_tot, -1)).any(axis=1)
    pool = np.flatnonzero(finite)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(pool, size=min(args.n_ref, pool.size), replace=False))
    # Pass every stored key (subsampled) so the augmentation chain has vcirc_kms/vterm_kms/sigma_z
    # etc.; the Gaia observation-model augmentations generate the masks/magnitudes as in training.
    ref_flat = {k: np.asarray(raw[k])[idx] for k in raw.files}
    ref_flat["j"] = np.asarray(raw["j"], float).reshape(n_tot, 1)[idx]
    print(f"Augmenting {len(idx)} training rows for the reference sim_summary ...", flush=True)
    # preprocessing first (mask_vcirc_radii 50->49, log10, drop_nan; train_val_split is a no-op in
    # transform mode), then the training-time augmentation chain -- exactly the evaluate() order.
    ref_flat = sim_pipe.transform(ref_flat)
    ref_flat = apply_augmentations_once(ref_flat, cs, sim_pipe, int(cs.seed))
    ref = np.asarray(ref_flat["sim_summary"], float)
    ref = ref.reshape(ref.shape[0], -1)
    ref_j = np.asarray(ref_flat["j"]).reshape(-1).astype(int)

    # --- observed: the real Gaia group (as evaluate_real builds it) ---
    real_pipe = build_pipeline(cr.preprocessing)
    real_pipe.load(os.path.join(md, PREPROCESSING_STATE))
    flat_real, m = _prepare_real_members(cr)
    flat_real = real_pipe.transform(flat_real)
    flat_real = apply_augmentations_once(flat_real, cr, real_pipe, int(cr.seed))
    obs = np.asarray(flat_real["sim_summary"], float)
    obs = obs.reshape(obs.shape[0], -1)
    obs_j = np.asarray(flat_real["j"]).reshape(-1).astype(int)

    print(f"sim_summary reference {ref.shape} (j counts {np.unique(ref_j, return_counts=True)[1]}) | "
          f"observed {obs.shape} j={obs_j}", flush=True)

    # names
    try:
        from hydrabflow.simulators.registry import get_simulator
        streams = getattr(get_simulator(cs.simulator), "target_streams", None) or {}
        jname = {int(v): str(k) for k, v in streams.items()}
    except Exception:
        jname = {}

    res = mmd_test(obs, ref, obs_j, ref_j, num_null=args.num_null,
                   rng=np.random.default_rng(args.seed))
    pm = per_member_scores(obs, ref, obs_j, ref_j)
    per_member = {jname.get(j, f"member_{j}"): round(v["percentile"], 1) for j, v in pm.items()}

    print(f"\nSTREAM channel (sim_summary) vs {ref.shape[0]} training streams:")
    print(f"  mmd_observed = {res['mmd_observed']:.4f}")
    print(f"  p_plain      = {res['p_value_plain']:.4f}")
    print(f"  p_stratified = {res.get('p_value_stratified')}")
    print(f"  per-stream Mahalanobis percentile: {per_member}")

    out_dir = args.out or args.real_run
    os.makedirs(out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from bayesflow.diagnostics.plots import mmd_hypothesis_test
        null = res.get("null_stratified", res.get("null_plain"))
        fig = mmd_hypothesis_test(mmd_null=np.asarray(null), mmd_observed=float(res["mmd_observed"]))
        fig.suptitle(f"Stream channel (sim_summary) MMD vs {ref.shape[0]} training streams\n"
                     f"p_strat={res.get('p_value_stratified')}  per-stream pct {per_member}",
                     fontsize=9)
        png = os.path.join(out_dir, "mmd_stream_vs_training.png")
        fig.savefig(png, dpi=140, bbox_inches="tight")
        print(f"\nSaved {png}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot failed: {exc}")

    payload = {"n_reference": int(ref.shape[0]), "n_features": int(ref.shape[1]),
               "mmd_observed": float(res["mmd_observed"]),
               "p_value_plain": res["p_value_plain"],
               "p_value_stratified": res.get("p_value_stratified"),
               "per_member_percentile": per_member}
    with open(os.path.join(out_dir, "mmd_stream_vs_training.json"), "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
