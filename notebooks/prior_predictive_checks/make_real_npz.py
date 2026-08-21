"""
Write one real disk's observation as an ``evaluate``-ready ``.npz``.

``hydrabflow-evaluate data.real_data_path=<npz>`` replays the fitted preprocessing and samples the
posterior, so the ``.npz`` has to carry exactly the keys the protoplan adapter reads.
``_realdisk.build_real_batch`` already assembles them (regridded images, log-interpolated SED, the
beam/noise/geometry encodings) -- this adds the three things a *posterior* run needs and the
prior-predictive checks do not:

1. **Both discrete branches.**  The estimator is ``p(theta | data, has_cavity)``, and a real disk does
   not come with that indicator, so every row is emitted twice -- ``has_cavity`` 0 then 1.  Two
   conditional posteriors, not one guess; `evaluate` writes `posterior_corner_obs0/1.png`.
2. **A placeholder for a band the disk lacks.**  Every adapter key must exist.  A missing band gets a
   zero image and the longest-wavelength measured band's conditions (in-prior, and log-safe -- three
   of them are log-transformed by the adapter, so zeros would be ``-inf``).  Pass the branch to
   ``eval.mask_condition_groups`` and none of it reaches the density.
3. **Zeros for the targets.**  The real path never scores them, but ``load_approximator``'s
   rebuild-from-config fallback probes the adapter, which concatenates them into
   ``inference_variables``.  Shape ``(N, 1)`` like the training cache: as ``(N,)`` the adapter
   concatenates 7 parameters into 14 columns and the weight load fails with a misleading
   "expected 2 variables, but received 0".

    uv run python notebooks/prior_predictive_checks/make_real_npz.py oph163131 /path/to/real.npz

Prints the ``eval.mask_condition_groups`` value the evaluate call needs.
"""

import pathlib
import sys

import itertools

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _realdisk as R  # noqa: E402

#: ALMA channel index -> the condition group name `GroupedFlowMatching` knows it by.
BRANCH_NAME = {0: "b9_input", 1: "b7_input", 2: "b6_input"}
PLACEHOLDER_KEYS = ("sigma_alma", "beam_maj", "beam_min", "beam_pa", "beam_pa_sin", "beam_pa_cos")


def build(disk: str, overrides=("experiment=protoplan",)) -> tuple[dict, list[str]]:
    """`(dataset, mask_groups)` -- the rows to save and the branches to mask when sampling."""
    cfg = R.load_cfg(overrides)
    train_lams = np.asarray(R.load_training_data(cfg)["sed_lams"][0, :, 0], dtype=np.float64)
    # The *randomized* training prior's ALMA sampling, not the disk's own beams: the network was
    # trained on that grid, so the real image must be regridded onto it (see §8 of the write-up).
    alma_px = R.randomized_alma_obs_px(cfg)
    batch, _aux = R.build_real_batch(disk, train_lams, alma_px=alma_px)

    present = max(R.disk_config(disk)["alma_channels"])
    missing = [j for j in range(len(R.ALMA_CHANNELS)) if f"im_jy_alma_{j}" not in batch]
    for j in missing:
        batch[f"im_jy_alma_{j}"] = np.zeros_like(batch[f"im_jy_alma_{present}"])
        for k in PLACEHOLDER_KEYS:
            batch[f"{k}_{j}"] = batch[f"{k}_{present}"].copy()

    for p in cfg.adapter.inference_variables:
        batch[p] = np.zeros((1, 1), dtype=np.float32)

    # One row per combination of the discrete indicators the adapter conditions on: a real disk
    # comes with none of them, so every branch is sampled and read as its own conditional
    # posterior. `[has_cavity]` gives the original two rows, 0 then 1; `[sil_id, has_cavity]`
    # gives four, in `itertools.product` order -- (0,0), (0,1), (1,0), (1,1).
    discrete = list(cfg.adapter.inference_conditions)
    combos = list(itertools.product([0.0, 1.0], repeat=len(discrete)))
    data = {k: np.concatenate([v] * len(combos), axis=0) for k, v in batch.items()}
    for i, name in enumerate(discrete):
        data[name] = np.array([c[i] for c in combos], dtype=np.float32)
    return data, [BRANCH_NAME[j] for j in missing]


if __name__ == "__main__":
    # make_real_npz.py <disk> <out.npz> [hydra override ...]
    # The overrides must be the ones the model was *trained* with -- they decide which targets get
    # placeholders and which indicators get a row.
    disk, out = sys.argv[1], sys.argv[2]
    overrides = tuple(sys.argv[3:]) or ("experiment=protoplan",)
    data, mask = build(disk, overrides)
    np.savez(out, **data)
    rows = [k for k in ("sil_id", "has_cavity") if k in data]
    print(f"\n{out}: {len(data)} keys, {len(data[rows[0]])} rows over {rows}")
    for i in range(len(data[rows[0]])):
        print(f"  obs{i}: " + ", ".join(f"{k}={data[k][i]:.0f}" for k in rows))
    print(f"pass to evaluate:  'eval.mask_condition_groups={mask}'".replace("'", "\"", 0))
