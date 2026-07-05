"""Cross-model posterior tension report (offline analysis helper — not a Hydra run stage).

Quantifies whether several trained models tell the same story about the SAME data: for each
shared parameter, the pairwise tension

    z = |median_i - median_j| / sqrt(std_i^2 + std_j^2)

between the posteriors saved by different runs. Run it once on real-data posteriors
(``evaluate_real``) and once on a shared simulated test set (``evaluate``): models that agree on
simulation but disagree on real data indicate misspecification-driven extrapolation rather than
training instability.

Usage:
    python scripts/report_cross_model_tension.py LABEL=PATH [LABEL=PATH ...]

``PATH`` is a posterior ``.npz`` or a run dir containing one (``posterior.npz``,
``compositional_posterior.npz`` or ``base_posterior.npz`` — first match wins). Bare paths get
their run-dir name as label. Arrays may be ``(n_datasets, n_samples, 1)`` or flat: with several
datasets the report shows the tension distribution across datasets.
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np

_CANDIDATES = ("posterior.npz", "compositional_posterior.npz", "base_posterior.npz")


def _resolve(path: str) -> str:
    if path.endswith(".npz"):
        return path
    for name in _CANDIDATES:
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No posterior .npz found under {path} (tried {_CANDIDATES})")


def _load_clean(path: str) -> dict[str, np.ndarray]:
    """Load a posterior .npz as {param: (n_datasets, n_samples)} float arrays."""
    out = {}
    with np.load(_resolve(path)) as data:
        for k in data.files:
            arr = np.asarray(data[k], dtype=float)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            if arr.ndim == 1:
                arr = arr[None, :]
            out[k] = arr  # (n_datasets, n_samples)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1

    runs: dict[str, dict[str, np.ndarray]] = {}
    for arg in argv:
        label, _, path = arg.rpartition("=")
        if not label:
            path = arg
            label = os.path.basename(os.path.normpath(os.path.dirname(_resolve(arg))
                                                      if arg.endswith(".npz") else arg))
        runs[label] = _load_clean(path)

    params = sorted(set.intersection(*(set(r) for r in runs.values())))
    n_datasets = min(next(iter(r.values())).shape[0] for r in runs.values())
    labels = list(runs)
    print(f"models: {labels}")
    print(f"shared parameters: {params}")
    print(f"datasets compared: {n_datasets}\n")

    if n_datasets == 1:
        print(f"{'parameter':32s} " + " | ".join(f"{lb}: median+/-std" for lb in labels))
        for p in params:
            cells = []
            for lb in labels:
                s = runs[lb][p][0]
                cells.append(f"{lb}: {np.median(s):.4g}+/-{s.std():.2g}")
            print(f"{p:32s} " + " | ".join(cells))
        print(f"\n{'parameter':32s} {'max z':>7s} {'median z':>9s}   worst pair")
        for p in params:
            zs = {}
            for a, b in combinations(labels, 2):
                sa, sb = runs[a][p][0], runs[b][p][0]
                zs[(a, b)] = abs(np.median(sa) - np.median(sb)) / np.sqrt(
                    sa.std() ** 2 + sb.std() ** 2
                )
            (wa, wb), wz = max(zs.items(), key=lambda kv: kv[1])
            print(f"{p:32s} {max(zs.values()):7.2f} {np.median(list(zs.values())):9.2f}   "
                  f"{wa} vs {wb} ({wz:.2f})")
    else:
        print(f"{'parameter':32s} {'median z':>9s} {'p90 z':>7s} {'p99 z':>7s} {'max z':>7s}"
              "   (pairwise, across datasets)")
        for p in params:
            all_z = []
            for a, b in combinations(labels, 2):
                sa, sb = runs[a][p][:n_datasets], runs[b][p][:n_datasets]
                z = np.abs(np.median(sa, axis=1) - np.median(sb, axis=1)) / np.sqrt(
                    sa.std(axis=1) ** 2 + sb.std(axis=1) ** 2
                )
                all_z.append(z)
            all_z = np.concatenate(all_z)
            print(f"{p:32s} {np.median(all_z):9.2f} {np.percentile(all_z, 90):7.2f} "
                  f"{np.percentile(all_z, 99):7.2f} {all_z.max():7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
