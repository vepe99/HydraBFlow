"""Split the fused-summary MMD misspecification test into its modality slices.

For a 2-modality fusion model with `head: null` the summary network output is the plain
concatenation [stream backbone | rotation-curve backbone], so the saved `summaries.npz` of an
`evaluate` real-data run (observed) and its `eval.misspecification_reference` run (reference) can
be re-tested slice by slice with the SAME Schmitt+21 machinery (`pipeline.misspecification`):

  * full        — the test as run by evaluate (stratified-by-j null), for reference
  * stream      — dims [0, d_stream): per-member stream summary, stratified null
  * curve       — dims [d_stream, D): the curve is GROUP-level, so it is tested at ONE ROW PER
                  POTENTIAL (observed 1 row vs one member per reference group); the members' copies
                  differ only by the per-member noise draw, and treating them as members would
                  triplicate the observed curve and force p -> 0.
  * shuffled    — full vector with the curve slice of each observed member replaced by that of a
                  random reference row: "stream as observed, curve typical" (and vice versa)
                  -> tells whether the flag needs the COMBINATION.

Usage:
  python scripts/misspecification_summary_slices.py --real-run <eval_real dir> \
      [--sim-run <eval_sim dir>] --d-stream 32 [--num-null 500]
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
    ap.add_argument("--sim-run", default=None, help="default: eval.misspecification_reference of the real run")
    ap.add_argument("--d-stream", type=int, required=True, help="summary_dim of the stream backbone")
    ap.add_argument("--num-null", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from hydrabflow.pipeline.misspecification import mmd_test, per_member_scores, _bf_mmd

    cfg = OmegaConf.load(os.path.join(args.real_run, ".hydra", "config.yaml"))
    sim_run = args.sim_run or str(cfg.eval.misspecification_reference)
    obs = np.load(os.path.join(args.real_run, "summaries.npz"))
    ref = np.load(os.path.join(sim_run, "summaries.npz"))
    O, oj = obs["summaries"], obs["j"].reshape(-1).astype(int)
    R, rj = ref["summaries"], ref["j"].reshape(-1).astype(int)
    ds, D = args.d_stream, O.shape[1]
    names = {int(v): str(k) for k, v in cfg.simulator.params.target_streams.items()}
    rng = np.random.default_rng(args.seed)
    out, nulls = {}, {}

    def run(tag, o, r, oj_, rj_, strat=True):
        res = mmd_test(o, r, oj_ if strat else None, rj_ if strat else None, num_null=args.num_null,
                       rng=np.random.default_rng(args.seed))
        pm = per_member_scores(o, r, oj_, rj_) if oj_ is not None else {}
        key = "null_stratified" if "null_stratified" in res else "null_plain"
        nulls[tag] = res[key]
        out[tag] = {
            "mmd": res["mmd_observed"], "p_plain": res["p_value_plain"],
            "p_stratified": res.get("p_value_stratified"),
            "null_median": float(np.median(res[key])), "null_95": float(np.quantile(res[key], 0.95)),
            "per_member_percentile": {names.get(k, k): round(v["percentile"], 1) for k, v in pm.items()},
        }
        print(f"{tag:12s} mmd {res['mmd_observed']:.3f}  null med/95 {np.median(res[key]):.3f}/"
              f"{np.quantile(res[key], 0.95):.3f}  p_plain {res['p_value_plain']:.3f}  "
              f"p_strat {res.get('p_value_stratified')}  members {out[tag]['per_member_percentile']}")

    run("full", O, R, oj, rj)
    run("stream", O[:, :ds], R[:, :ds], oj, rj)

    # curve: one row per potential. Reference groups are stored member-major (j = 0,1,2,0,1,2,...).
    m = len(np.unique(rj))
    Rc = R[::m, ds:]                       # first member of every group
    Oc = O[:1, ds:]
    res = mmd_test(Oc, Rc, num_null=args.num_null, rng=np.random.default_rng(args.seed))
    # Mahalanobis percentile of the observed curve summary among the reference potentials
    mu, cov = Rc.mean(0), np.cov(Rc, rowvar=False) + 1e-6 * np.eye(Rc.shape[1])
    inv = np.linalg.pinv(cov)
    d_ref = np.sqrt(np.einsum("ij,jk,ik->i", Rc - mu, inv, Rc - mu))
    d_obs = float(np.sqrt(((Oc - mu) @ inv @ (Oc - mu).T).item()))
    nulls["curve"] = res["null_plain"]
    out["curve"] = {"mmd": res["mmd_observed"], "p_plain": res["p_value_plain"],
                    "null_median": float(np.median(res["null_plain"])),
                    "null_95": float(np.quantile(res["null_plain"], 0.95)),
                    "note": "one row per potential (observed 1 vs %d reference)" % len(Rc),
                    "mahalanobis_percentile": float((d_ref <= d_obs).mean() * 100)}
    print(f"{'curve':12s} mmd {res['mmd_observed']:.3f}  null med/95 {out['curve']['null_median']:.3f}/"
          f"{out['curve']['null_95']:.3f}  p_plain {res['p_value_plain']:.3f}  "
          f"Mahalanobis pct {out['curve']['mahalanobis_percentile']:.1f} (1 row vs {len(Rc)} potentials)")

    # combination tests: swap one slice of the observed trio for a typical reference slice
    pools = {j: np.flatnonzero(rj == j) for j in np.unique(oj)}
    def swapped(keep_stream: bool, n_rep=20):
        mm, pp = [], []
        for _ in range(n_rep):
            o = O.copy()
            if keep_stream:   # observed stream + a typical curve (one reference potential)
                g = rng.integers(len(Rc)); o[:, ds:] = Rc[g]
            else:             # typical streams (matched j) + observed curve
                for i, j in enumerate(oj):
                    o[i, :ds] = R[pools[j][rng.integers(len(pools[j]))], :ds]
            mm.append(_bf_mmd(o, R))
        return float(np.median(mm)), float(np.mean(np.asarray(mm)[:, None] <= nulls["full"][None, :]))
    for tag, ks in (("obs_stream+typ_curve", True), ("typ_stream+obs_curve", False)):
        med, p = swapped(ks)
        out[tag] = {"mmd_median_over_swaps": med, "p_vs_full_stratified_null": p}
        print(f"{tag:22s} mmd {med:.3f}  p (vs full stratified null) {p:.3f}")

    outdir = args.out or args.real_run
    json.dump(out, open(os.path.join(outdir, "misspecification_summary_slices.json"), "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, tag in zip(axes, ("full", "stream", "curve")):
        ax.hist(nulls[tag], bins=40, color="0.6", label="bootstrap null")
        ax.axvline(out[tag]["mmd"], color="C3", lw=2, label=f"observed (p={out[tag].get('p_stratified', out[tag]['p_plain']):.3f})")
        ax.set_title({"full": f"full fused summary ({D} dims)", "stream": f"stream slice (0:{ds})",
                      "curve": f"curve slice ({ds}:{D}), 1 row / potential"}[tag])
        ax.set_xlabel("MMD"); ax.legend(fontsize=8)
    fig.suptitle("Summary-space misspecification test split by modality"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "misspecification_summary_slices.png"), dpi=140)
    print("saved", os.path.join(outdir, "misspecification_summary_slices.{json,png}"))


if __name__ == "__main__":
    main()
