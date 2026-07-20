#!/usr/bin/env python
"""Posterior-predictive check of the stream-frame summary tracks — median AND per-bin std
(the "too cold streams" check), the posterior twin of ``scripts/ppc_summary_statistics.py``.

Takes a completed ``evaluate_real composition=global`` run directory. It REUSES the saved
posterior draws (no network re-sampling): the pooled global posterior ``posterior.npz`` (saved in
the model's native — possibly log10 — space) and, with ``--source per-stream``, the per-member
``single_stream_posterior.npz``. Draws are mapped back to physical units by inverting the run's
preprocessing ``log10_transform`` keys. At the global level the LOCAL (progenitor phase-space)
parameters are not inferred, so each global draw is paired with locals sampled from the
simulator's per-stream priors — the standard "posterior on globals x prior on nuisance locals"
predictive (those priors are observationally pinned to the clusters' measured kinematics).

Each (draw, stream) pair is re-simulated with the run's own simulator (CPU/joblib — spray or
restricted N-body, whatever the run used), and the resulting streams are rendered with the SAME
track + per-bin-std overlay against the real Gaia members, so prior- and posterior-predictive
figures are directly comparable.

Unlike ppc_rotation_curve.py this cannot be standalone (it needs the forward model), so it
imports hydrabflow — GPU probing is disabled up front; runs fine on a CPU-only box.

Usage:
  uv run python scripts/ppc_posterior_summary_statistics.py \
      --run-dir outputs/.../eval_real --n-samples 40 [--source per-stream]
"""

from __future__ import annotations

import argparse
import os
import sys

# CPU-only: stop autocvd GPU probing in every joblib worker + silence AGAMA chatter. Must be set
# before the hydrabflow import below.
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _SCRIPTS)

from ppc_summary_statistics import render  # noqa: E402  (same-dir import, like ppc_ancillary)

DEFAULT_REAL = os.path.join(_REPO, "assets", "gaia",
                            "gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")


def _load_posterior(path):
    d = np.load(path, allow_pickle=True)
    return {k: np.asarray(d[k]).reshape(np.asarray(d[k]).shape[0], -1) for k in d.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="evaluate_real composition=global run dir")
    ap.add_argument("--n-samples", type=int, default=40,
                    help="posterior draws per stream reused from the saved posterior")
    ap.add_argument("--source", choices=["pooled", "per-stream"], default="pooled",
                    help="pooled = the compositional global posterior for every stream; "
                    "per-stream = each member's own single-stream posterior")
    ap.add_argument("--real", default=DEFAULT_REAL)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None, help="figure path (default: <run-dir>/"
                    "ppc_posterior_summary_statistics_<source>.png)")
    ap.add_argument("--noise", action="store_true",
                    help="apply the run's training observation model (Gaia window/subsample/DR3 "
                    "noise/vlos mask) to the re-simulated streams before binning — fair "
                    "'too cold' comparison against the real per-bin std")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from hydrabflow.pipeline.compositional import log10_keys_from_pipeline
    from hydrabflow.preprocessing.registry import build_pipeline
    from hydrabflow.simulators.registry import get_simulator
    from hydrabflow.simulators.stream_common import sample_prior_value

    cfg = OmegaConf.load(os.path.join(args.run_dir, ".hydra", "config.yaml"))
    sim = get_simulator(cfg.simulator)
    log10_keys = set(log10_keys_from_pipeline(build_pipeline(cfg.preprocessing)))

    if args.source == "pooled":
        post = _load_posterior(os.path.join(args.run_dir, "posterior.npz"))
        group_of = lambda s: 0  # noqa: E731  — one pooled group for every stream
    else:
        post = _load_posterior(os.path.join(args.run_dir, "single_stream_posterior.npz"))
        group_of = lambda s: s  # noqa: E731  — member s conditions member s's simulation

    # posterior.npz is saved in the model's native space: invert the preprocessing log10 keys.
    for key in set(post) & log10_keys:
        post[key] = 10.0 ** post[key]

    priors_global = sim._priors_global
    priors_local = sim._priors_local
    streams = sorted(sim.target_streams.items(), key=lambda kv: kv[1])  # (name, j) in j order
    m = len(streams)

    n_draws = next(iter(post.values())).shape[1]
    n = min(args.n_samples, n_draws)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_draws, size=n, replace=False)

    inferred = [k for k in priors_global if k in post]
    fixed = [k for k, spec in priors_global.items()
             if k not in post and spec["type"] == "identity"]
    missing = [k for k, spec in priors_global.items()
               if k not in post and spec["type"] != "identity"]
    if missing:
        print(f"WARNING: non-identity globals absent from the posterior, drawn from the PRIOR "
              f"instead: {missing}")

    print(f"Posterior-predictive summary tracks | source={args.source} | reusing {n} saved "
          f"draws x {m} streams = {n * m} re-simulated streams ({type(sim).__name__})")
    print(f"  globals from posterior: {inferred}")
    print(f"  identity constants:     {fixed}")
    print("  locals from per-stream priors (not inferred at composition=global)")

    # Flat rows, draw-major: row i*m + s = (draw idx[i], stream s) -> reshape (n, m, P, 6) below.
    flat: dict[str, np.ndarray] = {}
    for key, spec in priors_global.items():
        if key in post:
            per_stream = np.stack(
                [post[key][group_of(s), idx] for s in range(m)], axis=1)  # (n, m)
            flat[key] = per_stream.reshape(n * m, 1)
        elif spec["type"] == "identity":
            flat[key] = np.full((n * m, 1), float(spec["prior_parameters"][0]))
        else:  # warned above: absent from the posterior, fall back to the prior
            flat[key] = np.repeat(sample_prior_value(spec, n, rng), m, axis=0)
    local_names = sorted({k for name, _ in streams for k in priors_local[name]})
    for key in local_names:
        cols = np.concatenate(
            [sample_prior_value(priors_local[name][key], n, rng) for name, _ in streams], axis=1
        )  # (n, m)
        flat[key] = cols.reshape(n * m, 1)

    out = sim.simulate(flat, np.random.default_rng(args.seed + 1))
    grouped = out["sim_data_projected"].reshape(n, m, -1, 6)

    nan_rows = int(np.isnan(grouped).all(axis=(2, 3)).sum())
    if nan_rows:
        print(f"  {nan_rows}/{n * m} re-simulated streams are NaN (failed rows), skipped in bins")

    sim_npz = os.path.join(args.run_dir, f"ppc_posterior_streams_{args.source}.npz")
    np.savez(sim_npz, sim_data_projected=grouped,
             **{k: v.reshape(n, m) for k, v in flat.items()})
    print(f"  re-simulated streams + parameters saved to {sim_npz}")

    sim_masks = None
    if args.noise:
        # Use the model's TRAINING augmentation chain (the eval_real run's own augmentation node
        # is the real-data preset, which expects observed keys) — read it from model_dir.
        from ppc_summary_statistics import augment_sim

        aug_cfg = None
        model_dir = str(cfg.get("model_dir", "") or "")
        if model_dir:
            mdp = model_dir if os.path.isabs(model_dir) else os.path.join(_REPO, model_dir)
            train_cfg_path = os.path.join(mdp, ".hydra", "config.yaml")
            if os.path.exists(train_cfg_path):
                root = OmegaConf.load(train_cfg_path)
                aug_cfg = OmegaConf.create(OmegaConf.to_container(root.augmentation, resolve=True))
        if aug_cfg is None:
            print("WARNING: training config not found under model_dir; using the default "
                  "stream_global_ibata_grid observation model")
        jgrid = np.tile(np.array([jj for _, jj in streams], dtype=float), (n, 1))
        grouped, attn, vmask = augment_sim(grouped, jgrid, seed=args.seed, aug_cfg=aug_cfg)
        sim_masks = (attn, vmask)

    fig = args.out or os.path.join(
        args.run_dir, f"ppc_posterior_summary_statistics_{args.source}.png")
    render(args.real, grouped, fig, n_sim=n, seed=args.seed,
           kind=f"Posterior-predictive ({args.source}"
           + (", noise-convolved)" if args.noise else ")"),
           sim_masks=sim_masks)


if __name__ == "__main__":
    main()
