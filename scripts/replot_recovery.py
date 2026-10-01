"""Replot base_/compositional_recovery.png of a composition=global sim-eval run with LaTeX labels.

Posteriors come from the run's saved {base,compositional}_posterior.npz; truths are re-read from
the test set through the model's fitted preprocessing (same path as evaluate), everything mapped
back to physical units. Parameter order/labels = scripts/replot_global_corner.PARAMS.

    python scripts/replot_recovery.py EVAL_SIM_DIR
"""

import os
import sys

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import bayesflow as bf  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from hydrabflow.pipeline import io  # noqa: E402
from hydrabflow.pipeline.adapter import fill_stream_grid_from_simulator  # noqa: E402
from hydrabflow.pipeline.compositional import flatten_members  # noqa: E402
from hydrabflow.registry import build_pipeline  # noqa: E402
from replot_global_corner import PARAMS  # noqa: E402

KEYS = [p[0] for p in PARAMS]
LABELS = [p[1] for p in PARAMS]


def load_physical(run_dir):
    """{mode: (estimates, targets)} for base/compositional, both in physical units."""
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    fill_stream_grid_from_simulator(cfg)  # saved configs are pre-fill
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(cfg.model_dir, "preprocessing_state.npz"))
    test = pipeline.transform(
        io.load_dataset(os.path.join(cfg.data.data_dir, cfg.eval.test_dataset_name))
    )
    n, m = np.asarray(test["j"]).shape[:2]
    targets = {
        "base": {k: np.asarray(v) for k, v in flatten_members(test, m).items() if k in KEYS},
        "compositional": {k: np.asarray(test[k]) for k in KEYS},
    }
    out = {}
    for mode, targ in targets.items():
        post = dict(np.load(os.path.join(run_dir, f"{mode}_posterior.npz")))
        assert np.asarray(post[KEYS[0]]).shape[0] == np.asarray(targ[KEYS[0]]).shape[0], mode
        est = pipeline.inverse_transform({k: post[k] for k in KEYS})
        targ = pipeline.inverse_transform(dict(targ))
        out[mode] = ({k: est[k] for k in KEYS}, {k: targ[k] for k in KEYS})
    return out


def main():
    run_dir = sys.argv[1]
    for mode, (est, targ) in load_physical(run_dir).items():
        fig = bf.diagnostics.recovery(
            estimates=est, targets=targ, variable_keys=KEYS, variable_names=LABELS,
        )
        path = os.path.join(run_dir, f"{mode}_recovery_latex.png")
        fig.savefig(path, bbox_inches="tight")
        print(path)


if __name__ == "__main__":
    main()
