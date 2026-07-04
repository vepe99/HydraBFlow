"""Structured (dataclass) config schemas for every Hydra config group.

The whole pipeline is configured through these typed dataclasses rather than ``_target_``
instantiation. Each group dataclass is registered in Hydra's :class:`ConfigStore` as
``base_<group>``; the concrete YAML files under ``conf/<group>/`` inherit it via their own
``defaults`` list and fill in values. Factory functions (``networks.factory``,
``simulators.registry``, ``preprocessing.registry``, ``augmentation.registry``,
``pipeline.adapter``) read these dataclasses to build the actual objects.

Adding a new variant of a component = a new YAML file in the right group (and, for simulators or
custom networks, a registered class). No change to this schema or to infrastructure code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING

# --------------------------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------------------------- #


@dataclass
class SimulatorConfig:
    """Forward model selection. ``name`` is resolved through the simulator registry.

    ``params`` is a free-form mapping passed verbatim to the simulator constructor, so each
    simulator can declare whatever it needs (number of particles, integrator settings, prior
    bounds, ...) without touching this schema. Parameter names and observable keys are reported
    by the simulator class itself (``BaseSimulator.parameter_names`` / ``observable_keys``).
    """

    name: str = MISSING
    params: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------- #
# Model (summary network + inference network)
# --------------------------------------------------------------------------------------------- #


@dataclass
class SummaryNetworkConfig:
    """Hyperparameters for a single BayesFlow summary network.

    ``type`` is resolved through the summary-network registry in ``networks.factory`` (shipped:
    set_transformer, deep_set, time_series_transformer; custom builders self-register via
    ``@register_summary_network``). Block-wise architectures (SetTransformer,
    TimeSeriesTransformer) are expanded to ``num_blocks``-length tuples from the scalar fields
    below. ``params`` is a free-form mapping for custom builders' extra hyperparameters.
    Multi-observable fusion is a documented extension point (see the factory) and is not wired
    by default.
    """

    type: str = "set_transformer"  # resolved through the summary-network registry
    summary_dim: int = 32
    num_blocks: int = 2
    num_heads: int = 4
    embed_dim: int = 64
    mlp_depth: int = 2
    mlp_width: int = 128
    dropout: float = 0.05
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceNetworkConfig:
    """Hyperparameters for the posterior (inference) network.

    ``type`` is resolved through the inference-network registry in ``networks.factory`` (shipped:
    flow_matching, diffusion; custom builders self-register via ``@register_inference_network``).
    ``params`` is a free-form mapping for custom builders' extra hyperparameters.
    """

    type: str = "flow_matching"  # resolved through the inference-network registry
    mlp_depth: int = 4
    mlp_width: int = 128
    dropout: float = 0.05
    time_embedding_dim: int = 32  # used by the diffusion network
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = MISSING
    summary_network: SummaryNetworkConfig = field(default_factory=SummaryNetworkConfig)
    inference_network: InferenceNetworkConfig = field(default_factory=InferenceNetworkConfig)


# --------------------------------------------------------------------------------------------- #
# Data (dataset generation + on-disk layout)
# --------------------------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    data_dir: str = "data"
    n_simulations: int = 10000
    chunk_size: int = 1000
    # Interpolated against n_simulations in YAML, e.g. "training_data_${data.n_simulations}.npz".
    dataset_name: str = MISSING
    real_data_path: Optional[str] = None


# --------------------------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------------------------- #


@dataclass
class TrainingConfig:
    n_epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    validation_fraction: float = 0.1
    # Keys the BayesFlow workflow standardizes internally (separate from the preprocessing module).
    standardize: List[str] = field(
        default_factory=lambda: ["inference_variables", "summary_variables"]
    )
    verbose: int = 2


# --------------------------------------------------------------------------------------------- #
# Preprocessing (deterministic, whole-dataset, applied once) vs Augmentation (per-batch)
# --------------------------------------------------------------------------------------------- #


@dataclass
class PreprocessingConfig:
    """Ordered list of preprocessing steps. Each entry is ``{name: <registry key>, ...params}``."""

    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AugmentationConfig:
    """Ordered list of per-batch augmentation names resolved through the augmentation registry."""

    steps: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------- #
# Composition (hierarchical global/local score modeling)
# --------------------------------------------------------------------------------------------- #


@dataclass
class CompositionConfig:
    """Compositional score modeling: train/evaluate the global or the local level separately.

    ``level``:
      * ``none`` (default) — single-level inference via ``bf.BasicWorkflow`` (unchanged).
      * ``global`` — infer the simulator's ``global_parameter_names`` conditioned on its
        ``context_keys``; uses ``bf.CompositionalWorkflow`` so evaluation can pool the
        exchangeable group members (e.g. streams) with compositional sampling.
      * ``local`` — infer ``local_parameter_names`` conditioned on the global parameters +
        ``context_keys``; evaluation on real data draws globals from a previously computed
        global posterior (ancestral sampling).

    ``global_run_dir`` points local-level evaluation at the *global* model's evaluation run
    (the directory containing its saved posterior) for ancestral sampling.
    """

    level: str = "none"
    global_run_dir: Optional[str] = None


# --------------------------------------------------------------------------------------------- #
# Adapter (BayesFlow structural transform)
# --------------------------------------------------------------------------------------------- #


@dataclass
class AdapterConfig:
    """Maps raw dataset keys to BayesFlow roles.

    ``inference_variables`` are concatenated into the inference target; ``summary_variables`` are
    fed to the summary network; ``inference_conditions`` are direct (non-summarized) conditions;
    ``drop`` removes unused keys. With a single observable, ``summary_variables`` has one entry;
    multiple entries are the seam for fusion (see ``pipeline.adapter``).

    Left empty (the default), ``inference_variables`` / ``summary_variables`` are derived from
    the selected simulator's ``parameter_names`` / ``observable_keys``, so a registered simulator
    needs no adapter config at all. Set them explicitly to override (e.g. to train on data that
    was not generated by a registered simulator — see docs/bring_your_own_data.md).
    """

    inference_variables: List[str] = field(default_factory=list)
    summary_variables: List[str] = field(default_factory=list)
    inference_conditions: List[str] = field(default_factory=list)
    drop: List[str] = field(default_factory=list)
    # Dataset/batch key holding a boolean attention mask for set-valued observables (e.g. produced
    # by an observational-window augmentation). It is renamed to BayesFlow's
    # ``summary_attention_mask`` role, which the approximator forwards to the summary network.
    attention_mask_key: Optional[str] = None


# --------------------------------------------------------------------------------------------- #
# Inference (posterior sampling) and Evaluation
# --------------------------------------------------------------------------------------------- #


@dataclass
class InferenceConfig:
    num_samples: int = 1000
    batch_size: int = 256


@dataclass
class EvalConfig:
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


# --------------------------------------------------------------------------------------------- #
# Hyperparameter tuning (Optuna)
# --------------------------------------------------------------------------------------------- #


@dataclass
class TuningConfig:
    study_name: str = "hydrabflow_study"
    storage_dir: str = "data/tuning"
    n_trials: int = 50
    directions: List[str] = field(default_factory=lambda: ["minimize", "minimize"])
    n_epochs: int = 10  # short training budget per trial
    # name -> {type: int|float|categorical, low, high, step, log, choices}
    search_space: Dict[str, Any] = field(default_factory=dict)
    # Persist every trial (trained model + posterior + diagnostic plots) and the shared, fit-once
    # preprocessing state under `artifacts_dir`. Keyed by the (study-global) Optuna trial number so
    # several processes pointing at the same study cooperatively fill one directory.
    save_artifacts: bool = True
    artifacts_dir: str = "${tuning.storage_dir}/${tuning.study_name}"


# --------------------------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------------------------- #


@dataclass
class RootConfig:
    seed: int = 42
    # Directory of a completed training run (holds the saved approximator + preprocessing state).
    # Required by evaluate / evaluate_real; set it to the timestamped train output dir.
    model_dir: Optional[str] = None
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)


def register_configs() -> None:
    """Register all schemas in Hydra's ConfigStore.

    Must be called before ``@hydra.main`` / ``compose`` so the YAML files (which inherit
    ``base_<group>`` via their ``defaults`` lists) validate against the dataclasses above.
    """
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=RootConfig)
    cs.store(group="simulator", name="base_simulator", node=SimulatorConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="model/summary_network", name="base_summary_network", node=SummaryNetworkConfig)
    cs.store(
        group="model/inference_network",
        name="base_inference_network",
        node=InferenceNetworkConfig,
    )
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="preprocessing", name="base_preprocessing", node=PreprocessingConfig)
    cs.store(group="augmentation", name="base_augmentation", node=AugmentationConfig)
    cs.store(group="adapter", name="base_adapter", node=AdapterConfig)
    cs.store(group="composition", name="base_composition", node=CompositionConfig)
    cs.store(group="inference", name="base_inference", node=InferenceConfig)
    cs.store(group="eval", name="base_eval", node=EvalConfig)
    cs.store(group="tuning", name="base_tuning", node=TuningConfig)
