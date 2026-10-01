"""Why does the pooled (compositional) real-data posterior sit away from the per-stream ones?

Gaussian bookkeeping of the masked compositional identity used at evaluation,

    p(theta | s_1..s_m, c)  ∝  p(theta)^(1-(m+1)) * prod_j p(theta | s_j) * p(theta | c),

in the network's parameter space (log10 on Sigma/r/z_Disk, as posterior.npz is stored). Needs the
three real-data sample sets of one run dir:
    eval_real/                     posterior.npz (pooled Global) + single_stream_posterior.npz (stream+curve)
    eval_real_only_sim_summary/    single_stream_posterior.npz (curve masked: stream only)
    eval_real_only_vcirc_kms/      single_stream_posterior.npz (streams masked: curve only; member 0)
and the prior from the training set's parameter columns.

Reports, per parameter: means of every piece, the Gaussian-predicted pool vs the sampled Global,
and each piece's information gain over the prior along the prior-whitened principal directions --
where a piece carries ~no information, the (1-n) prior division is ill-conditioned and the pooled
mean is set by small differences between near-prior pieces.

    python scripts/diagnose_compositional_pooling.py RUN_DIR [--train-npz TRAINING.npz]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

LOG10 = {"Sigma_Disk", "r_Disk", "z_Disk"}
STREAMS = ["Pal5", "NGC3201", "M68"]


def gauss(x):
    return x.mean(0), np.cov(x, rowvar=False)


def stack(npz, keys, i):
    return np.column_stack([np.asarray(npz[k])[i].reshape(-1) for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--train-npz", default="data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_ou24vc_hydrabflow/training_data_300000.npz")
    ap.add_argument("--n-prior", type=int, default=100000)
    a = ap.parse_args()
    er = os.path.join(a.run_dir, "eval_real")
    glob = np.load(os.path.join(er, "posterior.npz"))
    keys = list(glob.files)
    both = np.load(os.path.join(er, "single_stream_posterior.npz"))
    s_only = np.load(os.path.join(a.run_dir, "eval_real_only_sim_summary", "single_stream_posterior.npz"))
    c_only = np.load(os.path.join(a.run_dir, "eval_real_only_vcirc_kms", "single_stream_posterior.npz"))

    tr = np.load(a.train_npz)
    prior = np.column_stack([np.asarray(tr[k]).reshape(-1)[: a.n_prior] for k in keys])
    for i, k in enumerate(keys):
        if k in LOG10:
            prior[:, i] = np.log10(prior[:, i])
    m0, S0 = gauss(prior)
    L0 = np.linalg.inv(S0)

    G = stack(glob, keys, 0)
    Sj = [stack(s_only, keys, i) for i in range(3)]
    Bj = [stack(both, keys, i) for i in range(3)]
    C = stack(c_only, keys, 0)

    # Gaussian pool of the pieces the sampler actually combines: 3 stream-only + 1 curve-only, prior^(1-4)
    pieces = [gauss(x) for x in Sj + [C]]
    Lp = sum(np.linalg.inv(S) for _, S in pieces) - (len(pieces) - 1) * L0
    hp = sum(np.linalg.inv(S) @ m for m, S in pieces) - (len(pieces) - 1) * L0 @ m0
    ev = np.linalg.eigvalsh(Lp)
    pool_ok = bool(ev.min() > 0)
    mp = np.linalg.solve(Lp, hp) if pool_ok else np.full(len(keys), np.nan)
    sp = np.sqrt(np.diag(np.linalg.inv(Lp))) if pool_ok else np.full(len(keys), np.nan)

    # self-consistency: p(theta|s_j,c) should equal p(theta|s_j) p(theta|c) / p(theta)
    Lc, mc = np.linalg.inv(gauss(C)[1]), gauss(C)[0]
    pred_both = []
    for x in Sj:
        m, S = gauss(x)
        L = np.linalg.inv(S) + Lc - L0
        pred_both.append(np.linalg.solve(L, np.linalg.inv(S) @ m + Lc @ mc - L0 @ m0))

    sd0 = np.sqrt(np.diag(S0))
    rows = {}
    print(f"{'param':34s} {'prior':>8s} " + " ".join(f"{'s:'+s[:4]:>8s}" for s in STREAMS)
          + f" {'curve':>8s} {'pool(G)':>8s} {'Global':>8s}   (means; shifts in prior sd below)")
    for i, k in enumerate(keys):
        ms = [x[:, i].mean() for x in Sj]
        print(f"{k:34s} {m0[i]:8.3f} " + " ".join(f"{v:8.3f}" for v in ms)
              + f" {C[:, i].mean():8.3f} {mp[i]:8.3f} {G[:, i].mean():8.3f}")
        rows[k] = dict(prior_mean=m0[i], prior_sd=sd0[i], stream_only=ms, curve_only=C[:, i].mean(),
                       stream_plus_curve=[x[:, i].mean() for x in Bj],
                       predicted_stream_plus_curve=[p[i] for p in pred_both],
                       gaussian_pool=mp[i], gaussian_pool_sd=sp[i], global_mean=G[:, i].mean(),
                       global_sd=G[:, i].std())
    print("\nGlobal - Gaussian pool, in Global sd:  "
          + "  ".join(f"{k.split('_')[0]}={(G[:, i].mean() - mp[i]) / G[:, i].std():+.2f}" for i, k in enumerate(keys)))
    print("\nself-consistency: sampled (stream+curve) minus predicted p(s)p(c)/p0, in prior sd")
    for j, s in enumerate(STREAMS):
        d = (Bj[j].mean(0) - pred_both[j]) / sd0
        print(f"  {s:8s} " + "  ".join(f"{k.split('_')[0]}={v:+.2f}" for k, v in zip(keys, d)))

    # information of each piece over the prior along prior-whitened directions: eig of S0^1/2 L S0^1/2
    w, V = np.linalg.eigh(S0)
    R = V @ np.diag(np.sqrt(w)) @ V.T
    print("\nprecision / prior precision along the piece's own principal directions (1 = no information):")
    info = {}
    for name, x in zip([f"s:{s}" for s in STREAMS] + ["curve"], Sj + [C]):
        g = np.sort(np.linalg.eigvalsh(R @ np.linalg.inv(gauss(x)[1]) @ R))
        info[name] = g.tolist()
        print(f"  {name:10s} " + " ".join(f"{v:7.2f}" for v in g))
    print("  pooled Lp eigenvalues (whitened): " + " ".join(f"{v:7.2f}" for v in np.sort(np.linalg.eigvalsh(R @ Lp @ R))))

    out = os.path.join(a.run_dir, "compositional_pooling_diagnostic.json")
    json.dump(dict(params=rows, info_gain=info, pool_positive_definite=pool_ok), open(out, "w"), indent=2,
              default=float)
    print(f"\n{out}")


if __name__ == "__main__":
    main()
