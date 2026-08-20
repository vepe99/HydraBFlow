"""Stage 3: posterior inference on held-out data, simulated or real.

Loads the approximator + fitted preprocessing from ``cfg.model_dir``, then samples. Two modes, one
code path: the simulated test set (has ground truth -> truth-aware diagnostics), or
``data.real_data_path=<your.npz>`` (no truth, no resimulation -> posterior pair plots only).

``model_dir`` is read-only: results go to this launch's own Hydra run dir (``get_run_dir()``), not
back into ``model_dir``, so one model can be evaluated any number of times without clobbering.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from hydrabflow import registry
from hydrabflow.pipeline import artifacts, io
from hydrabflow.pipeline._app import make_cli
from hydrabflow.pipeline.workflow import build_workflow
from hydrabflow.registry import build_pipeline
from hydrabflow.utils.paths import PREPROCESSING_STATE, get_run_dir
from hydrabflow.utils.seed import seed_everything

log = logging.getLogger(__name__)


def observed_condition_mask(network, drop, n_rows: int) -> np.ndarray:
    """``(n_rows, sum(group_sizes))``, 0 on every column of a group named in ``drop``.

    This is the modality-ablation path: BayesFlow already honours ``observed_condition_mask`` as a
    ``sample`` kwarg (it is sliced per batch and repeated alongside the conditions), so marking a
    modality unobserved at inference needs no change to the network -- only the column layout,
    which the inference network carries as ``group_names``/``group_sizes``.
    """
    names = list(getattr(network, "group_names", []) or [])
    sizes = list(getattr(network, "group_sizes", []) or [])
    if not names or len(names) != len(sizes):
        raise ValueError(
            "eval.mask_condition_groups needs an inference network exposing group_names and "
            f"group_sizes (i.e. GroupedFlowMatching, model.inference_network.params."
            f"missing_modality_prob > 0); got names={names!r} sizes={sizes!r}. Note a model "
            "trained without modality dropout has never seen a masked condition vector, so "
            "masking one would be extrapolation even if the layout were known.")
    unknown = [d for d in drop if d not in names]
    if unknown:
        raise ValueError(f"Unknown condition group(s) {unknown}; choose from {names}")

    mask = np.ones((n_rows, sum(sizes)), dtype="float32")
    offset = 0
    for name, size in zip(names, sizes):
        if name in drop:
            mask[:, offset:offset + size] = 0.0
        offset += size
    log.info("Masking condition group(s) %s -> %d of %d condition columns marked unobserved",
             list(drop), int((mask[0] == 0).sum()), mask.shape[1])
    return mask


def run_evaluation(cfg):
    seed_everything(cfg.seed)
    run_dir = get_run_dir()
    if not cfg.model_dir:
        raise ValueError(
            "evaluate requires `model_dir` to point at a completed training run, e.g. "
            "model_dir=outputs/<sim>/<run>/<timestamp>"
        )

    # 1. Load the data and replay the *fitted* preprocessing (no re-fit, no split).
    real = bool(cfg.data.real_data_path)
    pipeline = build_pipeline(cfg.preprocessing)
    pipeline.load(os.path.join(cfg.model_dir, PREPROCESSING_STATE))
    if real:
        path = cfg.data.real_data_path
        raw = io.load_dataset(path)
    else:
        # `io.load_config_dataset` honours a simulator that owns its data on disk, in which case
        # `eval.test_dataset_name` does not exist and the held-out rows come from the
        # preprocessing split -- the same rows training held out, by construction.
        path = os.path.join(cfg.data.data_dir, cfg.eval.test_dataset_name)
        if os.path.exists(path):
            raw = io.load_dataset(path)
        else:
            _, raw = build_pipeline(cfg.preprocessing).fit_transform(
                io.load_config_dataset(cfg), np.random.default_rng(cfg.seed))
            path = f"held-out split of {cfg.simulator.name}"
    data = pipeline.transform(raw)

    # 2. Augmentations, once, with a fixed draw -- exactly as train does to its validation split.
    #    They are training-time in the sense of being stochastic and per-batch, but they are part of
    #    the *observation model*: an augmentation that adds instrument noise, or that splits one
    #    array into per-instrument keys, produces the very inputs the adapter reads. Skipping them
    #    here would hand the network a differently-shaped batch (or a noise-free one), so the
    #    posterior would be sampled off-distribution.
    #    Real data has already been observed, so nothing is applied to it.
    if not real:
        for aug in registry.build_augmentations(
                cfg.augmentation, np.random.default_rng(cfg.seed),
                context={"pipeline": pipeline}):
            data = aug(data)
        data = {k: np.asarray(v) for k, v in data.items()}

    # 3. Rebuild the workflow and load the trained approximator into it. The batch is passed as a
    #    probe so a full-model load that fails can fall back to rebuild + load_weights.
    workflow = build_workflow(cfg)
    workflow.approximator = artifacts.load_approximator(
        cfg.model_dir, workflow=workflow,
        probe_batch={k: v[:2] for k, v in data.items()})

    # 4. Sample the posterior for every observation, then score it if the truth is known.
    #    `eval.mask_condition_groups` marks whole modalities unobserved, which is how a run
    #    measures what one instrument was contributing -- and how a real source that simply lacks
    #    a band (Oph 163131 has no 880 um map) is evaluated at all.
    from bayesflow.utils import MaskName

    sample_kwargs = {}
    drop = list(cfg.eval.mask_condition_groups)
    if drop:
        n_rows = len(next(iter(data.values())))
        sample_kwargs[MaskName.OBSERVED_CONDITION] = observed_condition_mask(
            workflow.approximator.inference_network, drop, n_rows)

    posterior = workflow.sample(
        num_samples=int(cfg.eval.num_samples),
        conditions=data,
        batch_size=int(cfg.eval.batch_size),
        **sample_kwargs,
    )
    param_names = list(cfg.adapter.inference_variables)
    artifacts.save_posterior(posterior, run_dir)
    # Corner plots either way: prior box behind, median marked, plus the truth when there is one.
    # A simulated test set has thousands of rows, so only the first few get a figure.
    truth = None if real else np.column_stack([np.asarray(data[p]).reshape(-1) for p in param_names])
    artifacts.save_posterior_plot(
        posterior, param_names, run_dir, truth=truth,
        bounds=artifacts.load_prior_bounds(cfg.model_dir, param_names),
        max_obs=None if real else 5)
    if not real:
        artifacts.run_diagnostics(cfg, posterior, data, param_names, run_dir)

    log.info("Inference on %s complete. Artifacts in %s", path, run_dir)
    return posterior


cli = make_cli(run_evaluation)


if __name__ == "__main__":
    cli()
