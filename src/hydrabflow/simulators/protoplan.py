"""The protoplanetary-disk RT dataset, as a simulator that owns its data.

The forward model here is a radiative-transfer code that is not runnable in-process: its output
is a fixed set of ~49k (parameters, clean image, clean SED) rows, cached as per-key `.npy` files.
So this class implements the two optional `BaseSimulator` hooks -- `load_dataset` and
`build_adapter` -- and `sample_prior`/`simulate` raise.  `hydrabflow-simulate` is not part of this
project's workflow.

What turns a clean RT row into a mock JWST+ALMA observation is the `protoplan_instrument`
augmentation, applied per batch during training.  That is where the beam and the noise levels
live, and they are plain `augmentation.params` config knobs.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np

from hydrabflow.registry import register_simulator
from hydrabflow.simulators._protoplan_spec import (
    DISCRETE_CONDITION_PARAMS,
    NPE_TARGET_PARAMS,
    build_adapter,
    filter_and_jitter_params,
)
from hydrabflow.simulators.base import BaseSimulator

log = logging.getLogger(__name__)

#: The image cache to read, and the two SED caches.  NHWC because the adapter and the CNNs want
#: channels-last; the precomputed NHWC file rather than the NCHW one plus a transpose here, since
#: that chains a full-file load, a boolean-index copy and a transpose copy -- up to 3x peak RAM.
DEFAULT_IMAGE_FILE = "im_jy_combined_NHWC.npy"


@register_simulator("protoplan")
class ProtoplanetaryDiskSimulator(BaseSimulator):
    """Reader for the cached RT dataset.  See `conf/simulator/protoplan.yaml` for the knobs."""

    parameter_names = list(NPE_TARGET_PARAMS)
    #: The raw keys the instrument augmentation consumes.  `im_jy` is split into `im_jy_jwst` +
    #: `im_jy_alma_{0,1,2}` by the augmentation, so it does not appear in the adapter's groups.
    observable_keys = ["im_jy", "im_lams", "seds", "sed_lams"]

    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        raise NotImplementedError(
            "The protoplanetary-disk forward model is an external radiative-transfer code; this "
            "simulator only reads its cached output. Skip `hydrabflow-simulate` and point "
            "`simulator.params.data_path` at the `split_combined/` cache directory."
        )

    simulate = sample_prior

    # ── the two BaseSimulator hooks ──────────────────────────────────────────────────
    def load_dataset(self) -> Dict[str, np.ndarray]:
        """The cached RT rows as one dataset dict, row-aligned and filtered.

        The jitter RNG is seeded explicitly (`param_jitter_seed`, default 0): unseeded it falls
        through to the global `np.random` state, so two runs of the same config would see
        *different* `log_r_cav` draws on the ~25k no-cavity rows, confounding any A/B between
        them -- the arms would differ in their training targets, not just in the flag under test.
        """
        data_path = str(self.params["data_path"])
        seed = int(self.params.get("param_jitter_seed", 0))
        image_file = str(self.params.get("image_file", DEFAULT_IMAGE_FILE))

        targets, columns, keep = filter_and_jitter_params(
            data_path, rng=np.random.default_rng(seed))
        data = dict(columns)

        def _masked(filename, mmap_mode="r"):
            return np.load(os.path.join(data_path, filename), mmap_mode=mmap_mode)[keep]

        # mmap for the image cache so the boolean index streams only the kept rows off disk
        # instead of first materialising the whole array.
        data["im_jy"] = _masked(image_file)
        data["im_lams"] = _masked("im_lams_combined.npy", mmap_mode=None)

        sed = _masked("sed_flx_jy_combined.npy", mmap_mode=None)
        assert np.all(sed > 0), "non-positive SED fluxes survived the row filter"
        data["seds"] = sed.reshape(-1, sed.shape[-1], 1)
        sed_lams = _masked("sed_lams_combined.npy", mmap_mode=None)
        data["sed_lams"] = sed_lams.reshape(-1, sed_lams.shape[-1], 1)

        log.info(
            "Loaded %d RT rows from %s: %d targets + %s, im_jy %s, seds %s",
            int(keep.sum()), data_path,
            len(targets), DISCRETE_CONDITION_PARAMS, data["im_jy"].shape, data["seds"].shape,
        )
        return data

    def build_adapter(self, cfg):
        """The bespoke adapter (asinh images, per-instrument routed conditions, SED fusion).

        `cfg` is the root `AdapterConfig`; `inference_variables` and `inference_conditions` are
        read from it, so a run can infer a subset of the 17 targets and condition on a subset of
        the discrete indicators.  The image knees come from `simulator.params` because they must
        move with the configured noise level.
        """
        return build_adapter(
            params=list(cfg.inference_variables) or None,
            conditions=list(cfg.inference_conditions) or None,
            asinh_sigma_jwst=self.params.get("asinh_sigma_jwst"),
            asinh_sigma_alma=self.params.get("asinh_sigma_alma"),
        )
