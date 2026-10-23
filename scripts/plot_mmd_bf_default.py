"""MMD hypothesis test with BayesFlow's DEFAULT null (companion to mmd_hypothesis_test.png).

Reuses the summaries an `evaluate` run already saved: the observed members (`<eval_real>/summaries.npz`)
and the simulated test-set members (`<eval_sim>/summaries.npz`). The null is
`bf.diagnostics.metrics.bootstrap_comparison` with `maximum_mean_discrepancy` and its defaults (uniform
bootstrap of len(observed) reference rows, 100 draws) — i.e. what `summary_space_comparison` would give,
minus the re-summarizing. The pipeline's own plot uses the stream-stratified null instead.

    python scripts/plot_mmd_bf_default.py --ref <eval_sim>/summaries.npz --obs <eval_real>/summaries.npz \
        --out <eval_real>/mmd_hypothesis_test_bfdefault
"""

import argparse
import json
import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from bayesflow.diagnostics.metrics import bootstrap_comparison  # noqa: E402
from bayesflow.diagnostics.plots import mmd_hypothesis_test  # noqa: E402
from bayesflow.metrics.functional import maximum_mean_discrepancy  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True)
p.add_argument("--obs", required=True)
p.add_argument("--out", required=True, help="output path without extension")
p.add_argument("--num-null", type=int, default=100)  # bayesflow default
p.add_argument("--seed", type=int, default=2026)
a = p.parse_args()

np.random.seed(a.seed)  # bootstrap_comparison draws with the global numpy RNG
ref = np.load(a.ref)["summaries"]
obs = np.load(a.obs)["summaries"]
mmd_obs, null = bootstrap_comparison(obs, ref, maximum_mean_discrepancy, num_null_samples=a.num_null)
pval = float((null >= mmd_obs).mean())

fig = mmd_hypothesis_test(mmd_null=null, mmd_observed=mmd_obs)
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}", bbox_inches="tight")
res = {"mmd_observed": float(mmd_obs), "p_value": pval, "num_null": a.num_null, "seed": a.seed,
       "null_quantiles": {q: float(np.quantile(null, float(q))) for q in ("0.05", "0.5", "0.95")}}
with open(f"{a.out}.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
