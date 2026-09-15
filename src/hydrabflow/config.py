"""Typed config schema. ``conf/config.yaml`` fills these in; the factories read them.

Only ``RootConfig`` is registered with Hydra (as ``base_config``). Because every group is a typed
field of it, a group YAML like ``conf/simulator/two_moons.yaml`` is validated against its dataclass
just by being selected — no per-group ``base_*`` node.

Adding a component = a new YAML in the right group + a registered class. This file rarely changes.
The stream-specific fields (``composition``, ``adapter.attention_mask_key``, the ``eval``
compositional/misspecification knobs) all default to "off", so the plain single-level pipeline is
unaffected by them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING


@dataclass
class SimulatorConfig:
    """``name`` resolves through the simulator registry; ``params`` goes to its constructor."""

    name: str = MISSING
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryNetworkConfig:
    """``type`` resolves through the summary-network registry (``networks/factory.py``).

    Attention width is ``num_heads * embed_dim_per_head``, so it stays divisible by the head count
    for any value a tuner draws. ``params`` carries extra hyperparameters for custom builders (the
    ``fusion`` network reads its per-observable backbone specs from there).
    """

    type: str = "set_transformer"
    summary_dim: int = 32
    num_blocks: int = 2
    num_heads: int = 4
    embed_dim_per_head: int = 16
    mlp_depth: int = 2
    mlp_width: int = 128
    dropout: float = 0.05
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceNetworkConfig:
    """``type`` resolves through the inference-network registry (``networks/factory.py``)."""

    type: str = "flow_matching"
    mlp_depth: int = 4
    mlp_width: int = 128
    dropout: float = 0.05
    time_embedding_dim: int = 32  # diffusion only
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    # Labels the run dir (``run_name`` defaults to it). A whole-model preset under ``conf/model/``
    # (e.g. ``model=stream_fusion_model5``) sets it; otherwise it is "<summary>+<inference>".
    name: str = "${model.summary_network.type}+${model.inference_network.type}"
    summary_network: SummaryNetworkConfig = field(default_factory=SummaryNetworkConfig)
    inference_network: InferenceNetworkConfig = field(default_factory=InferenceNetworkConfig)


@dataclass
class DataConfig:
    data_dir: str = "data"
    n_simulations: int = 10000
    chunk_size: int = 1000
    dataset_name: str = MISSING
    real_data_path: Optional[str] = None


@dataclass
class TrainingConfig:
    n_epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 5e-4  # peak LR of BayesFlow's cosine schedule
    validation_fraction: float = 0.1
    standardize: List[str] = field(default_factory=lambda: ["inference_variables"])
    verbose: int = 2
    save_best_weights: bool = True


@dataclass
class PreprocessingConfig:
    """Ordered steps, each ``{name: <registry key>, ...params}``. Deterministic, applied once."""

    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AugmentationConfig:
    """Ordered augmentation names (registry keys) + the params they read. Stochastic, per batch."""

    steps: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionConfig:
    """Compositional score modeling: train/evaluate the global or the local level separately.

    ``level``:
      * ``none`` (default) — single-level inference via ``bf.BasicWorkflow``.
      * ``global`` — infer the simulator's ``global_parameter_names`` conditioned on its
        ``context_keys``; ``bf.CompositionalWorkflow`` pools the exchangeable group members
        (e.g. streams) with compositional sampling at evaluation.
      * ``local`` — infer ``local_parameter_names`` conditioned on the globals + ``context_keys``;
        real-data evaluation draws the globals from a saved global posterior (ancestral sampling).

    ``global_run_dir`` points a local-level real-data evaluation at the global model's evaluation
    run (the directory holding its ``posterior.npz``).
    """

    level: str = "none"
    global_run_dir: Optional[str] = None


@dataclass
class AdapterConfig:
    """Dataset keys -> BayesFlow roles. Empty lists derive from the simulator's declaration."""

    inference_variables: List[str] = field(default_factory=list)
    summary_variables: List[str] = field(default_factory=list)
    inference_conditions: List[str] = field(default_factory=list)
    drop: List[str] = field(default_factory=list)
    # Batch key holding a boolean per-element attention mask for a set-valued observable (e.g. from
    # an observational-window augmentation). Renamed to BayesFlow's ``summary_attention_mask``
    # role, which the approximator forwards to the summary network.
    attention_mask_key: Optional[str] = None


@dataclass
class EvalConfig:
    """Posterior sampling + diagnostics, used by the evaluate stage (simulated or real data)."""

    test_dataset_name: str = MISSING
    num_samples: int = 1000
    batch_size: int = 256
    diagnostics: List[str] = field(
        default_factory=lambda: [
            "metrics", "recovery", "calibration_ecdf", "coverage", "z_score_contraction"
        ]
    )
    # Free-form kwargs forwarded to the sampler — used by compositional sampling, e.g.
    # {method: two_step_adaptive, steps: adaptive, compositional_bridge_d1: 0.1667}.
    sample_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Summary-space misspecification test (real data, composition=global): an
    # `evaluate composition=global` run dir holding summaries.npz (or a direct .npz path).
    # Empty = skip the test.
    misspecification_reference: str = ""
    misspecification_num_null: int = 200
    # Compositional prior score (composition=global). "spec" = analytic simulator prior spec;
    # "kde" = Gaussian KDE fit on the training draws in the network's native (log10) space.
    prior_score: str = "spec"
    prior_kde_samples: str = ""
    prior_kde_max_points: int = 4096
    prior_kde_bandwidth: float = 0.0
    prior_kde_impl: str = "diagonal"  # diagonal (closed form) | jax (full covariance)
    # Compositional condition masking (composition=global, inference_network=grouped_diffusion).
    # `member_groups` = condition groups observed on each of the m member items; `extra_items` =
    # one further item per entry, listing the groups it observes. Masking the group-level
    # observables (the rotation curve) out of the members and giving them their own item is what
    # keeps each likelihood in the compositional product exactly once. Empty = stock behaviour.
    # Condition groups observed by ordinary (non-compositional) `sample` calls -- [sim_summary]
    # for one stream alone, [vcirc_kms] for the rotation curve alone. Empty = everything observed.
    observed_groups: List[str] = field(default_factory=list)
    member_groups: List[str] = field(default_factory=list)
    extra_items: List[List[str]] = field(default_factory=list)


@dataclass
class TuningConfig:
    study_name: str = "hydrabflow_study"
    storage_dir: str = "data/tuning"
    artifacts_dir: str = "${tuning.storage_dir}/${tuning.study_name}"
    save_artifacts: bool = True
    n_trials: int = 50
    n_epochs: int = 10
    directions: List[str] = field(default_factory=lambda: ["minimize", "minimize"])
    # dotted config path -> {type: int|float|categorical, low, high, step, log, choices}
    search_space: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RootConfig:
    seed: int = 42
    run_name: str = MISSING
    model_dir: Optional[str] = None  # a completed train run; required by evaluate
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)


def register_configs() -> None:
    """Register the root schema so ``conf/config.yaml`` validates against it."""
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(name="base_config", node=RootConfig)
