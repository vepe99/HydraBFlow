"""Export each fusion backbone of a trained run as ``<out>/<key>_weights.npz`` (for ``warm_weights`` /
``frozen_weights`` of MaskedFusionNetwork). The run's own ``.hydra`` config (or an eval subdir's) rebuilds
the workflow first, which registers the lazily-defined serializable classes keras needs to load it.

    JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0 .venv/bin/python scripts/export_backbone_weights.py \
        --run <trial dir> --config <trial dir>/eval_real/.hydra/config.yaml --out <dir>
"""
import argparse, os
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()

from omegaconf import OmegaConf
from hydrabflow.pipeline._bf_patches import apply_bayesflow_patches
from hydrabflow.pipeline.adapter import fill_adapter_from_simulator, fill_stream_grid_from_simulator
from hydrabflow.pipeline.artifacts import load_approximator
from hydrabflow.pipeline.workflow import build_workflow

cfg = OmegaConf.load(a.config); OmegaConf.set_struct(cfg, False)
fill_adapter_from_simulator(cfg); fill_stream_grid_from_simulator(cfg)
apply_bayesflow_patches(); build_workflow(cfg)
net = load_approximator(a.run).summary_network
os.makedirs(a.out, exist_ok=True)
for k, bb in net.backbones.items():
    np.savez(os.path.join(a.out, f"{k}_weights.npz"), *[np.asarray(w) for w in bb.get_weights()])
    print(k, len(bb.get_weights()), "arrays ->", os.path.join(a.out, f"{k}_weights.npz"))
