"""Typed config schema. ``conf/config.yaml`` fills these in; the factories read them.

Only ``RootConfig`` is registered with Hydra (as ``base_config``). Because every group is a typed
field of it, a group YAML like ``conf/simulator/two_moons.yaml`` is validated against its dataclass
just by being selected — no per-group ``base_*`` node, and no YAML inheriting from another YAML.

Adding a component = a new YAML in the right group + a registered class. This file rarely changes.
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
    for any value a tuner draws. ``params`` carries extra hyperparameters for custom builders.
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
class AdapterConfig:
    """Dataset keys -> BayesFlow roles. Empty lists derive from the simulator's declaration."""

    inference_variables: List[str] = field(default_factory=list)
    summary_variables: List[str] = field(default_factory=list)
    inference_conditions: List[str] = field(default_factory=list)
    drop: List[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    """Posterior sampling + diagnostics, used by the evaluate stage."""

    test_dataset_name: str = MISSING
    num_samples: int = 1000
    batch_size: int = 256
    diagnostics: List[str] = field(
        default_factory=lambda: [
            "metrics", "recovery", "calibration_ecdf", "z_score_contraction"
        ]
    )
    #: Condition groups to mark *unobserved* when sampling, by name (see the inference network's
    #: ``group_names``). This is the modality-ablation knob: the group's columns of the resolved
    #: condition vector are marked unobserved via BayesFlow's ``observed_condition_mask``, the same
    #: mechanism ``missing_modality_prob`` uses during training -- excluded from attention under
    #: ``subnet: diffusion_transformer``, zeroed under an MLP subnet. Requires an inference network that
    #: exposes ``group_names``/``group_sizes`` (i.e. ``GroupedFlowMatching``).
    mask_condition_groups: List[str] = field(default_factory=list)


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
    eval: EvalConfig = field(default_factory=EvalConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)


def register_configs() -> None:
    """Register the root schema so ``conf/config.yaml`` validates against it."""
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(name="base_config", node=RootConfig)
