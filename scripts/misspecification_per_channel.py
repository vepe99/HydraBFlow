"""Per-channel model-misspecification MMD test (localize WHERE the misspecification comes from).

The standard test (``pipeline/misspecification.py``) runs one MMD on the FUSED summary-network
output, so it says the observation is atypical but not which input drives it. This script runs the
SAME MMD hypothesis test (Schmitt+21, with the j-stratified bootstrap null and per-member
Mahalanobis percentiles from ``pipeline.misspecification``) SEPARATELY on each *projected
observable channel* the network is fed:

  * ``sim_summary``  -- the per-stream binned stream-frame summary statistics (the "stream" channel)
  * ``vcirc_kms``    -- the (log10, masked) model rotation curve
  * ``vterm_kms``    -- the HI terminal-velocity curve (ancillary)
  * ``sigma_z``      -- the local surface density (ancillary condition)

It works on the RAW projected observables (adapter + augmentation output), NOT the summary
embedding, so the result is model-free: it localizes the misspecification in *data* space rather
than through the (itself extrapolating) summary network. Both sides are built with the exact eval
pipelines (``_load_test_data`` + ``flatten_members`` for the sim reference, ``_prepare_real_members``
+ ``pipeline.transform`` for the observed group) and the same augmentation draw, after replaying the
CLI's ``fill_adapter_from_simulator`` / ``fill_stream_grid_from_simulator`` (the saved .hydra config
is pre-fill).

Per-member channels (stream identity varies across a group -> ``sim_summary``) use the j-stratified
null + per-stream Mahalanobis percentiles. Group-level channels (one value per potential -> the
rotation curve / ancillaries) are de-duplicated to one row per group and tested with the plain null
(observed group = 1 row vs the reference group cloud).

CPU only (forces JAX_PLATFORMS=cpu). Usage:
    uv run python scripts/misspecification_per_channel.py \
        --sim-run  outputs/ibata_onedisk_grid_m200c/default/eval_sim_333 \
        --real-run outputs/ibata_onedisk_grid_m200c/default/eval_real
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np


def _flat2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a.reshape(a.shape[0], -1)


def _per_member_safe(real, sim, j_real, j_sim, per_member_scores):
    """per_member_scores, but robust to scalar channels (F==1, e.g. sigma_z), where np.cov
    collapses to 0-d. For F==1 use the univariate standardized distance and its percentile."""
    real = np.asarray(real, float); sim = np.asarray(sim, float)
    if real.shape[1] > 1:
        return per_member_scores(real, sim, j_real, j_sim)
    obs_j = np.asarray(j_real).reshape(-1).astype(int)
    ref_j = np.asarray(j_sim).reshape(-1).astype(int)
    out = {}
    for i, j in enumerate(obs_j):
        pool = sim[ref_j == j, 0]
        if pool.size < 2:
            continue
        mu, sd = pool.mean(), pool.std() + 1e-12
        d_obs = abs(real[i, 0] - mu) / sd
        d_ref = np.abs(pool - mu) / sd
        out[int(j)] = {"mahalanobis": float(d_obs),
                       "reference_median": float(np.median(d_ref)),
                       "percentile": float((d_ref <= d_obs).mean() * 100.0),
                       "n_reference": int(pool.size)}
    return out


def _group_level_flags(test_data, m: int) -> dict:
    """Which raw test-set channels are group-level (one value per potential, shared by the m
    members: rotation curve / ancillaries)?  Uses ``flatten_members``' own rule on the GROUPED
    arrays — per-member iff ``ndim >= 3 and shape[1] == m`` — because a post-augmentation
    variance check fails: the noise augmentations run after flattening and give each of the m
    repeated copies an independent draw. Channels created later by the augmentation chain
    (``sim_summary``) are per-member by construction and simply absent here."""
    flags = {}
    for key, arr in test_data.items():
        a = np.asarray(arr)
        flags[key] = not (a.ndim >= 3 and a.shape[1] == m)
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-run", required=True, help="evaluate composition=global run dir (reference)")
    ap.add_argument("--real-run", required=True, help="evaluate_real composition=global run dir (observed)")
    ap.add_argument("--model-dir", default=None, help="train dir (default: sim-run's model_dir)")
    ap.add_argument("--channels", nargs="*", default=None,
                    help="channels to test (default: adapter summary_variables + sigma_z)")
    ap.add_argument("--num-null", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from hydrabflow.pipeline.adapter import (
        fill_adapter_from_simulator, fill_stream_grid_from_simulator,
    )
    from hydrabflow.pipeline.compositional import apply_augmentations_once, flatten_members
    from hydrabflow.pipeline.evaluate import _load_test_data
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
    md = args.model_dir or cs.get("model_dir") or os.path.join(
        os.path.dirname(args.sim_run), "train")

    # Sim reference: flat member rows + one augmentation draw (mirrors evaluate.py).
    test_data, pipeline = _load_test_data(cs, md)
    ctx = next(iter(cs.adapter.inference_conditions), "j")
    n, m = np.asarray(test_data[ctx]).shape[:2]
    group_flags = _group_level_flags(test_data, m)
    flat_sim = apply_augmentations_once(flatten_members(test_data, m), cs, pipeline, int(cs.seed))

    # Observed group (mirrors evaluate_real.py): its own preprocessing pipeline carries the
    # attach_observed_vcirc/vterm/sigma_z steps that inject the real MW curves, loaded from the
    # SAME fitted state as training.
    real_pipeline = build_pipeline(cr.preprocessing)
    real_pipeline.load(os.path.join(md, PREPROCESSING_STATE))
    flat_real, mr = _prepare_real_members(cr)
    flat_real = real_pipeline.transform(flat_real)
    flat_real = apply_augmentations_once(flat_real, cr, real_pipeline, int(cr.seed))

    j_sim = np.asarray(flat_sim["j"]).reshape(-1).astype(int)
    j_real = np.asarray(flat_real["j"]).reshape(-1).astype(int)

    # Human stream names.
    try:
        from hydrabflow.simulators.registry import get_simulator
        streams = getattr(get_simulator(cs.simulator), "target_streams", None) or {}
        jname = {int(v): str(k) for k, v in streams.items()}
    except Exception:
        jname = {}

    channels = args.channels or (list(cs.adapter.summary_variables) +
                                 [c for c in cs.adapter.inference_conditions if c != "j"])

    print(f"Per-channel misspecification MMD | sim n={n} m={m} | real m={mr} | "
          f"channels={channels} | num_null={args.num_null}")

    rng = np.random.default_rng(args.seed)
    results = {}
    for ch in channels:
        if ch not in flat_sim or ch not in flat_real:
            print(f"  [skip] {ch}: not present in sim/real flat")
            continue
        sim = _flat2d(flat_sim[ch])
        real = _flat2d(flat_real[ch])
        group_level = group_flags.get(ch, False)

        if group_level:
            # One row per group / per observed potential (the m flat copies are the same curve,
            # up to independent noise draws -- keep the first).
            sim_u = sim.reshape(n, m, -1)[:, 0, :]
            real_u = real.reshape(mr, -1)[0:1, :]
            res = mmd_test(real_u, sim_u, num_null=args.num_null, rng=rng)
            pm = _per_member_safe(real_u, sim_u, np.zeros(1), np.zeros(n), per_member_scores)
            per_member = {"MW": {"percentile": round(v["percentile"], 1),
                                 "mahalanobis": round(v["mahalanobis"], 3)}
                          for v in pm.values()}
            unit = "group"
        else:
            res = mmd_test(real, sim, j_real, j_sim, num_null=args.num_null, rng=rng)
            pm = _per_member_safe(real, sim, j_real, j_sim, per_member_scores)
            per_member = {jname.get(j, f"member_{j}"): {"percentile": round(v["percentile"], 1),
                                                        "mahalanobis": round(v["mahalanobis"], 3)}
                          for j, v in pm.items()}
            unit = "member"

        results[ch] = {
            "unit": unit,
            "n_features": int(sim.shape[1]),
            "mmd_observed": round(res["mmd_observed"], 4),
            "p_plain": res["p_value_plain"],
            "p_stratified": res.get("p_value_stratified"),
            "per_member_percentile": per_member,
            "_null_plain": res["null_plain"],
            "_null_strat": res.get("null_stratified"),
        }
        pm_str = " | ".join(f"{k} {v['percentile']:.0f}pct" for k, v in per_member.items())
        print(f"  {ch:12s} [{unit:6s} F={sim.shape[1]:3d}] mmd={res['mmd_observed']:.4f} "
              f"p_plain={res['p_value_plain']:.3f} "
              f"p_strat={res.get('p_value_stratified')} {pm_str}")

    # ---- plot: one null histogram per channel with the observed MMD marked ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chs = list(results)
    fig, axes = plt.subplots(1, len(chs), figsize=(4.4 * len(chs), 3.8), squeeze=False)
    axes = axes[0]
    for ax, ch in zip(axes, chs):
        r = results[ch]
        null = np.asarray(r["_null_strat"] if r["_null_strat"] is not None else r["_null_plain"])
        ax.hist(null, bins=40, color="0.7", edgecolor="none")
        ax.axvline(r["mmd_observed"], color="crimson", lw=2, label=f"obs {r['mmd_observed']:.3f}")
        p = r["p_stratified"] if r["p_stratified"] is not None else r["p_plain"]
        ax.set_title(f"{ch}\n({r['unit']}, p={p:.3f})", fontsize=9)
        ax.set_xlabel("MMD"); ax.legend(fontsize=7)
    fig.suptitle("Per-channel misspecification MMD (observed vs sim reference null)", y=1.03)
    fig.tight_layout()
    out = args.out or os.path.join(args.real_run, "misspecification_per_channel.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")

    payload = {ch: {k: v for k, v in r.items() if not k.startswith("_")} for ch, r in results.items()}
    payload["_meta"] = {"sim_run": args.sim_run, "real_run": args.real_run, "n": n, "m": m,
                        "num_null": args.num_null, "figure": out}
    with open(os.path.join(args.real_run, "misspecification_per_channel.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\n" + json.dumps(payload, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
