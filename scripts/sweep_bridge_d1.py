"""Sweep ``compositional_bridge_d1`` on the compositional (pooled) posterior of one trained model.

Same Hydra composition as ``hydrabflow.pipeline.evaluate`` at composition=global, but ONLY the
compositional stage (no base per-member sampling), the model loaded once, and one sub-directory
``d1_<value>/`` with ``compositional_{posterior.npz,metrics.json,calibration_ecdf.png,...}`` per
value. Values come from the env var ``D1_VALUES`` (comma-separated; default 1/6..1).

    D1_VALUES=0.1667,0.25,0.333,0.5,1 python scripts/sweep_bridge_d1.py <same overrides as evaluate>
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from hydrabflow.pipeline import artifacts
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.compositional import (
    apply_augmentations_once, apply_mask_plan, build_prior_score, condition_keys,
    flatten_members, group_members, log10_keys_from_pipeline, sample_kwargs,
)
from hydrabflow.pipeline.evaluate import _load_model, _load_test_data
from hydrabflow.registry import get_simulator
from hydrabflow.utils.paths import POSTERIOR_SAMPLES, get_run_dir
from hydrabflow.utils.seed import seed_everything

log = logging.getLogger(__name__)


def run(cfg):
    values = [float(v) for v in os.environ.get("D1_VALUES", "0.1667,0.25,0.333,0.5,1").split(",")]
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    workflow = _load_model(cfg)
    test_data, pipeline = _load_test_data(cfg, cfg.model_dir)
    context = next(iter(cfg.adapter.inference_conditions), "j")
    n, m = np.asarray(test_data[context]).shape[:2]
    flat = apply_augmentations_once(flatten_members(test_data, m), cfg, pipeline, int(cfg.seed))
    param_names = list(cfg.adapter.inference_variables)
    log10_keys = log10_keys_from_pipeline(pipeline)
    grouped = group_members(flat, n, m)
    conditions = {k: grouped[k] for k in condition_keys(cfg) if k in grouped}
    prior_score = build_prior_score(
        cfg, get_simulator(cfg.simulator), log10_keys=log10_keys, param_order=param_names,
        seed=int(cfg.seed),
    )
    conditions = apply_mask_plan(workflow, cfg, conditions, m)
    targets = pipeline.inverse_transform({k: np.asarray(test_data[k]) for k in param_names})

    summary_path = os.path.join(run_dir, "sweep_summary.json")
    summary = json.load(open(summary_path))["metrics"] if os.path.exists(summary_path) else {}
    for d1 in values:
        out = os.path.join(run_dir, f"d1_{d1:g}")
        os.makedirs(out, exist_ok=True)
        kw = sample_kwargs(cfg) | {"compositional_bridge_d1": d1}
        log.info("compositional_bridge_d1=%g: %d datasets, kwargs=%s", d1, n, kw)
        seed_everything(cfg.seed)
        posterior = workflow.compositional_sample(
            num_samples=int(cfg.eval.num_samples), conditions=conditions,
            compute_prior_score=prior_score, batch_size=int(cfg.eval.batch_size), **kw,
        )
        artifacts.save_posterior(posterior, out, name=f"compositional_{POSTERIOR_SAMPLES}")
        artifacts.run_diagnostics(
            cfg, pipeline.inverse_transform(dict(posterior)), targets, param_names, out,
            prefix="compositional_",
        )
        with open(os.path.join(out, "compositional_metrics.json")) as f:
            summary[f"{d1:g}"] = json.load(f)
        with open(summary_path, "w") as f:
            json.dump({"param_names": param_names, "metrics": summary}, f, indent=2)
    for d1, mtr in summary.items():
        log.info("d1=%s  rmse=%.4f  calib=%.4f  calib per param=%s", d1, mtr["rmse"]["mean"],
                 mtr["calibration_error"]["mean"],
                 [round(v, 3) for v in mtr["calibration_error"]["values"]])


cli = make_cli(run)

if __name__ == "__main__":
    cli()
