# Configuration

`conf/config.yaml` is the whole configuration, plus three swappable groups:

```
conf/
├── config.yaml
├── simulator/                 two_moons | multimodal
├── model/summary_network/     set_transformer | deep_set | time_series_transformer | fusion
└── model/inference_network/   flow_matching | diffusion
```

Select a group file by name, override any scalar by path:

```bash
uv run hydrabflow-train simulator=multimodal model/summary_network=deep_set \
    training.n_epochs=100 model.inference_network.mlp_width=256
```

Types are enforced by the dataclasses in `src/hydrabflow/config.py`, so a typo or wrong type fails
before anything runs. `-m` turns any override into a sweep: `-m training.batch_size=128,512`.

## Blocks

| Block           | Key knobs                                                                        |
| --------------- | -------------------------------------------------------------------------------- |
| top level       | `seed`, `run_name` (output label), `model_dir` (input: a completed train run)      |
| `data`          | `data_dir`, `n_simulations`, `chunk_size`, `dataset_name`, `real_data_path`       |
| `training`      | `n_epochs`, `batch_size`, `learning_rate`, `validation_fraction`, `standardize`, `save_best_weights` |
| `preprocessing` | ordered `steps`, each `{name: <registry key>, ...params}`                         |
| `augmentation`  | `steps` (registry names) + shared `params`                                       |
| `adapter`       | dataset keys → BayesFlow roles; leave empty to derive from the simulator          |
| `eval`          | `test_dataset_name`, `num_samples`, `batch_size`, `diagnostics`                   |
| `tuning`        | `study_name`, `n_trials`, `n_epochs`, `search_space`                              |

### `run_name` vs `model_dir`

They point in opposite directions — one names the run you are creating, the other points at a run you
already have.

**`run_name`** is an *output* label, used by every stage. It is interpolated into the output path
`outputs/${simulator.name}/${run_name}/<timestamp>/` and defaults to
`<summary_type>+<inference_type>` (e.g. `set_transformer+flow_matching`), so runs group by
architecture. Override it to label an experiment: `run_name=wider_summary`.

**`model_dir`** is an *input* path, `null` by default: the directory of a completed `train` run,
holding `approximator.keras` and `preprocessing_state.npz`. Only `evaluate` reads it, and it fails
without it — it needs the trained model *and* that run's fitted preprocessing state, so test or real
data is transformed exactly as the training data was.

```bash
uv run hydrabflow-train run_name=wider_summary
# -> outputs/two_moons/wider_summary/2026-08-05_14-02-11/

uv run hydrabflow-evaluate model_dir=outputs/two_moons/wider_summary/2026-08-05_14-02-11
# reads that dir, writes a new dir of its own (named by run_name again)
```

`evaluate` never writes into `model_dir`, so one trained model can be evaluated any number of times
without overwriting anything.

## Notes that actually bite

- **`training.learning_rate`** is the *peak* LR: BayesFlow wraps it in cosine decay with 5% warmup
  and trains with AdamW. Don't pass your own optimizer — it disables that schedule.
- **`training.standardize: [inference_variables]`** is BayesFlow's internal z-scoring. The
  observable side is standardized by the `standardize` preprocessing step instead, because that
  step's fitted state is saved and replayed at inference time.
- **Preprocessing order matters**: steps before `train_val_split` see the whole dataset; steps after
  it are fit on the train split only and applied to both. `transform` (test/real data) replays the
  fitted steps and skips the split.
- **Preprocessing vs augmentation**: preprocessing is deterministic and applied once to the whole
  dataset; augmentation is stochastic and applied per batch inside `fit_offline`.
- **`adapter`** is normally untouched. Set it explicitly only for data with no simulator, to infer a
  subset of parameters, or to pass extra keys as `inference_conditions`. More than one
  `summary_variables` key automatically builds a `FusionNetwork` (one backbone per key).

## Network groups

Both network configs share `mlp_depth`, `mlp_width`, `dropout` and a free-form `params` dict for
builder-specific knobs. Summary networks add `summary_dim`, `num_blocks`, `num_heads`, and
`embed_dim_per_head` — attention width is `num_heads * embed_dim_per_head`, so it stays divisible
by the head count for any value a tuner draws.

`fusion.yaml` additionally uses `params.backbones: {<observable key>: <type>}` to give one
observable a different backbone than `type`.
