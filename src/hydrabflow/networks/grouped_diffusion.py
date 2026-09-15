"""Modality-coherent condition masking for the diffusion inference network.

Ported from the protoplan branch's ``GroupedFlowMatching`` (``networks/protoplan.py``) with the
base class swapped to ``DiffusionModel`` -- compositional sampling exists only on the diffusion
model -- and extended with *per-compositional-item* masks, which is what the stream project needs.

Why the per-item masks
----------------------
``flatten_members`` (``pipeline/compositional.py``) copies every *group-level* observable (the
rotation curve) into all ``m`` members, so each compositional item is ``(stream_j, curve)``. The
stock identity

    s(theta, t, Y) = (1-n)(1-t) grad log p(theta) + sum_j s(theta, t, y_j)
    <=>  p(theta|Y) ∝ p(theta)^(1-n) prod_j p(theta|y_j)

then multiplies the curve likelihood in ``n`` times instead of once, which narrows the posterior
by up to ``sqrt(n)`` in the directions the curve constrains.

With a mask plan the same trained network is evaluated as ``n+1`` items instead of ``n``::

    item 1..m : stream_j observed, curve masked
    item m+1  : curve observed, stream masked

so every likelihood factor appears exactly once. This is exact -- not an approximation -- as long
as the streams and the curve are conditionally independent given theta, which holds here: the
streams are integrated independently in the same potential and the curve is a deterministic
function of theta plus independently drawn observational noise.

The network must be *trained* with ``missing_modality_prob > 0`` for this to be valid: masked
conditioning sets have to be genuine amortised posteriors, not off-manifold extrapolations.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import bayesflow as bf
from keras import ops
from bayesflow.utils.masks import MaskName, random_mask
from bayesflow.utils.serialization import deserialize, serializable, serialize

from hydrabflow.registry import register_inference_network


#: Name of the synthetic leading group that absorbs ``adapter.inference_conditions`` -- everything
#: in the resolved condition vector that is not one of the summary backbones. Always observed.
CONDITIONS_GROUP = "_conditions"


@serializable("hydrabflow.networks")
class GroupedDiffusionModel(bf.networks.DiffusionModel):
    """``DiffusionModel`` with modality-coherent condition dropout and per-item masks.

    Parameters
    ----------
    group_sizes : sequence of int
        Widths of the *droppable* condition groups, in the order they appear in the resolved
        condition vector. For a ``MaskedFusionNetwork`` with ``head: null`` that is the
        per-backbone ``summary_dim`` in ``sorted(backbones)`` order -- a fusion head would mix
        the branches and make "drop a slice" meaningless, so it must be off.

        Anything the resolved vector carries *before* those groups (i.e.
        ``adapter.inference_conditions``) is absorbed into one always-observed leading group,
        whose width is inferred on first use as ``width - sum(group_sizes)``. Conditions are
        not modalities: dropping them asks the network to marginalise over something it is
        meant to be conditioned on.
    group_names : sequence of str
        Names in ``group_sizes`` order, so a mask can be addressed by modality name.
    missing_modality_prob : float
        Per-group, per-sample drop probability during training. ``0.0`` disables the masking
        and the class behaves exactly like ``DiffusionModel``.
    """

    def __init__(
        self,
        *args,
        group_sizes: Sequence[int],
        group_names: Sequence[str] = (),
        missing_modality_prob: float = 0.0,
        lead_width: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.group_sizes = [int(s) for s in group_sizes]
        self.group_names = [str(n) for n in group_names]
        self.missing_modality_prob = float(missing_modality_prob)
        # Resolved on first call from the actual condition width, then serialized so a
        # reloaded checkpoint does not have to see a training batch to know its layout.
        self.lead_width = None if lead_width is None else int(lead_width)
        if self.group_names and len(self.group_names) != len(self.group_sizes):
            raise ValueError(
                f"group_names has {len(self.group_names)} entries but group_sizes has "
                f"{len(self.group_sizes)}; they index the same groups."
            )
        #: Set by `set_item_mask_plan`; consumed by `compositional_score`.
        self._item_mask_plan: list[list[str]] | None = None
        #: Set by `set_observed_groups`; consumed by `score` for ordinary (non-compositional)
        #: sampling, where there is no item axis to hang a plan off.
        self._observed_groups: list[str] | None = None

    # ---------------------------------------------------------------- layout

    def _resolve_lead(self, width: int) -> int:
        """Width of the always-observed leading group, inferred from the condition vector."""
        if self.lead_width is None:
            lead = int(width) - sum(self.group_sizes)
            if lead < 0:
                raise ValueError(
                    f"group_sizes sums to {sum(self.group_sizes)} but the resolved condition "
                    f"vector is only {width} wide. group_sizes must list the summary "
                    "backbones' output widths (sorted by key, fusion head off)."
                )
            self.lead_width = lead
        return self.lead_width

    def _all_sizes(self, width: int) -> list[int]:
        lead = self._resolve_lead(width)
        return ([lead] if lead else []) + self.group_sizes

    def _all_names(self, width: int) -> list[str]:
        lead = self._resolve_lead(width)
        names = self.group_names or [f"group_{i}" for i in range(len(self.group_sizes))]
        return ([CONDITIONS_GROUP] if lead else []) + list(names)

    def group_mask(self, observed: Sequence[str], width: int) -> np.ndarray:
        """``(1, width)`` row that is 1 on the columns of every group named in *observed*.

        The always-observed leading group is on regardless -- masking it out is never what a
        modality plan means.
        """
        names, sizes = self._all_names(width), self._all_sizes(width)
        unknown = set(observed) - set(names)
        if unknown:
            raise ValueError(f"unknown condition group(s) {sorted(unknown)}; have {names}")
        keep = [n == CONDITIONS_GROUP or n in observed for n in names]
        return np.repeat(np.array([keep], dtype="float32"), sizes, axis=-1)

    def set_item_mask_plan(self, plan: Sequence[Sequence[str]] | None) -> None:
        """One list of observed group names per compositional item (or ``None`` to disable).

        ``[[ "sim_summary" ], [ "sim_summary" ], [ "sim_summary" ], [ "vcirc_kms" ]]`` turns
        ``m=3`` stream items plus one curve-only item into the de-duplicated factorisation.
        """
        self._item_mask_plan = None if plan is None else [list(p) for p in plan]

    def set_observed_groups(self, groups: Sequence[str] | None) -> None:
        """Condition groups observed by ordinary ``sample()`` calls (``None`` = all of them).

        The single-observation counterpart of a mask plan: ``["sim_summary"]`` asks the same
        trained network for p(theta | one stream), ``["vcirc_kms"]`` for p(theta | curve alone).
        Applied inside ``score`` rather than threaded through ``sample`` because the mask has to
        survive the approximator's ``num_samples`` repetition, which the caller does not control.
        """
        self._observed_groups = None if groups is None else list(groups)

    # ---------------------------------------------------------------- masking

    def observed_condition_mask(self, batch_size: int, width: int):
        """``(batch_size, width)``, 0 on every column of a dropped group.

        The draw covers the droppable groups only; the always-observed leading group is
        prepended afterwards. Drawing over all of them and forcing the leading one back to 1
        would let ``keep_one`` spend its guarantee on a group that was never at risk, leaving
        rows with every modality dropped -- pure-prior training samples.
        """
        lead = self._resolve_lead(width)
        keep = random_mask(
            (batch_size, len(self.group_sizes)),
            self.missing_modality_prob,
            keep_one=True,
            seed_generator=self.seed_generator,
        )
        if lead:
            ones = ops.ones_like(keep[:, :1])
            keep = ops.concatenate([ones, keep], axis=-1)
        return ops.repeat(keep, np.array(self._all_sizes(width)), axis=-1)

    def _apply(self, conditions, mask):
        """Zero the columns of *conditions* that *mask* marks unobserved.

        Only ``DiffusionTransformer`` consumes ``observed_condition_mask`` natively -- it
        tokenizes each condition scalar and drops those tokens out of attention, which is a
        strictly better "absent" than a zeroed value the model would still attend to. For any
        MLP-style subnet the mask is silently filtered out by ``_collect_mask_kwargs`` (its
        ``_subnet_mask_keys`` is empty), so "missing" has to be encoded the only way a flat
        vector can encode it: zero the group's slice. The two are exclusive, not cumulative.

        A mask whose batch dimension divides the condition one is tiled, so a plain ``sample``
        mask survives the ``num_samples`` repetition without the caller reshaping it.
        """
        if conditions is None or mask is None:
            return conditions
        if MaskName.OBSERVED_CONDITION in getattr(self, "_subnet_mask_keys", ()):
            return conditions
        mask = ops.cast(mask, ops.dtype(conditions))
        n_cond, n_mask = ops.shape(conditions)[0], ops.shape(mask)[0]
        if n_mask != n_cond and n_cond % n_mask == 0:
            mask = ops.tile(mask, (n_cond // n_mask, 1))
        return conditions * mask

    def compute_metrics(self, x, conditions=None, sample_weight=None, stage="training", **kwargs):
        if (
            stage == "training"
            and self.missing_modality_prob > 0
            and conditions is not None
            and MaskName.OBSERVED_CONDITION not in kwargs
        ):
            width = int(ops.shape(conditions)[-1])
            kwargs[MaskName.OBSERVED_CONDITION] = self.observed_condition_mask(
                ops.shape(conditions)[0], width
            )
        conditions = self._apply(conditions, kwargs.get(MaskName.OBSERVED_CONDITION))
        return super().compute_metrics(
            x, conditions=conditions, sample_weight=sample_weight, stage=stage, **kwargs
        )

    def score(self, xz, time=None, log_snr_t=None, conditions=None, training=False, **kwargs):
        """Sampling-time counterpart of the masking in ``compute_metrics``.

        ``velocity`` delegates here and so does the compositional score, so hooking this one
        method covers plain sampling, guided sampling and compositional sampling alike.
        """
        if (
            self._observed_groups is not None
            and conditions is not None
            and MaskName.OBSERVED_CONDITION not in kwargs
        ):
            # Tiled to the condition batch here rather than left to broadcast: the transformer
            # subnet indexes the mask per row, so a single (1, width) row would not line up.
            width = int(ops.shape(conditions)[-1])
            row = self.group_mask(self._observed_groups, width)
            kwargs[MaskName.OBSERVED_CONDITION] = ops.convert_to_tensor(
                np.repeat(row, int(ops.shape(conditions)[0]), axis=0)
            )
        conditions = self._apply(conditions, kwargs.get(MaskName.OBSERVED_CONDITION))
        return super().score(
            xz, time=time, log_snr_t=log_snr_t, conditions=conditions, training=training, **kwargs
        )

    # ------------------------------------------------------- compositional path

    def compositional_score(
        self, xz, time, conditions, seed=None, compute_prior_score=None,
        mini_batch_size=None, **kwargs,
    ):
        """Inject the per-item mask, then defer to the stock compositional score.

        Mini-batching is disabled while a plan is active: ``_compositional_score_direct``
        subsamples *items* with ``take_along_axis`` on ``conditions`` alone, so the mask rows
        would no longer line up with the items they belong to. It defaults to
        ``max(0.1 * n, 2)``, i.e. 2 of 3 items -- silently the wrong mask on every step.
        """
        if self._item_mask_plan is not None:
            batch, num_items = ops.shape(conditions)[:2]
            width = int(ops.shape(conditions)[-1])
            if len(self._item_mask_plan) != num_items:
                raise ValueError(
                    f"item mask plan has {len(self._item_mask_plan)} entries but there are "
                    f"{num_items} compositional items."
                )
            rows = [self.group_mask(p, width) for p in self._item_mask_plan]
            if compute_prior_score is None:
                # Upstream appends a zero-condition row to get the prior score from the
                # network itself; nothing is observed there.
                rows.append(np.zeros((1, width), dtype="float32"))
            plan = np.concatenate(rows, axis=0)[None, ...]  # (1, num_total, width)
            kwargs[MaskName.OBSERVED_CONDITION] = ops.convert_to_tensor(
                np.repeat(plan, int(batch), axis=0)
            )
            mini_batch_size = None
        return super().compositional_score(
            xz, time, conditions, seed=seed, compute_prior_score=compute_prior_score,
            mini_batch_size=mini_batch_size, **kwargs,
        )

    def _repeat_mask_kwargs_over_items(self, kwargs, num_total):
        """Flatten a rank-3 per-item mask the same way ``conditions`` is flattened.

        Upstream repeats a ``(batch, width)`` mask once per item, which is right for a mask
        that varies per *row*. A plan mask varies per *item*, and arrives as
        ``(batch, num_total, width)`` -- exactly the layout ``conditions`` has before its own
        ``reshape(batch * num_total, width)``, so it flattens identically and every row lines
        up with the item it masks.
        """
        plan = kwargs.get(MaskName.OBSERVED_CONDITION)
        if plan is not None and len(ops.shape(plan)) == 3:
            rest = {k: v for k, v in kwargs.items() if k != MaskName.OBSERVED_CONDITION}
            out = super()._repeat_mask_kwargs_over_items(rest, num_total)
            shape = ops.shape(plan)
            out[MaskName.OBSERVED_CONDITION] = ops.reshape(plan, (shape[0] * shape[1], shape[2]))
            return out
        return super()._repeat_mask_kwargs_over_items(kwargs, num_total)

    # ---------------------------------------------------------------- (de)serialization

    def get_config(self):
        return super().get_config() | serialize(
            {
                "group_sizes": self.group_sizes,
                "group_names": self.group_names,
                "missing_modality_prob": self.missing_modality_prob,
                "lead_width": self.lead_width,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


def _subnet_kwargs(cfg) -> tuple[str, dict]:
    """``(subnet name, subnet kwargs)`` from the inference-network config.

    ``params.subnet: diffusion_transformer`` is load-bearing for the masking: it is the only
    subnet that consumes ``observed_condition_mask`` natively, tokenizing each condition scalar
    and excluding the masked tokens from attention. With the default ``time_mlp`` the mask
    degrades to zeroing the group's columns (see ``GroupedDiffusionModel._apply``).
    """
    params = cfg.params or {}
    subnet = str(params.get("subnet", "time_mlp"))
    if subnet != "diffusion_transformer":
        return subnet, {
            "widths": [int(cfg.mlp_width)] * int(cfg.mlp_depth),
            "time_embedding_dim": int(cfg.time_embedding_dim),
        }
    # Only the keys the run actually sets; the rest keep BayesFlow's DiffusionTransformer
    # defaults. `dt_width`/`dt_num_layers` are scalars so an Optuna search space can address them.
    kwargs: dict = {}
    if "dt_width" in params and "dt_num_layers" in params:
        kwargs["widths"] = [int(params["dt_width"])] * int(params["dt_num_layers"])
    for key, name, cast in (
        ("dt_num_heads", "num_heads", int),
        ("dt_expansion_factor", "expansion_factor", float),
        ("dt_time_embedding_dim", "time_embedding_dim", int),
        ("dt_dropout", "dropout", float),
    ):
        if key in params:
            kwargs[name] = cast(params[key])
    return subnet, kwargs


def _fusion_group_sizes(model_cfg) -> tuple[list[int], list[str]]:
    """Droppable group widths + names from the fusion summary network's backbones.

    ``MaskedFusionNetwork`` concatenates its backbones in ``sorted(keys)`` order, so that is the
    order the resolved condition vector carries them in. A fusion *head* would mix the branches
    and make "drop a slice" meaningless, so it must be off.
    """
    from hydrabflow.config import SummaryNetworkConfig
    from omegaconf import OmegaConf

    summary = model_cfg.summary_network
    params = summary.params or {}
    backbones = params.get("backbones")
    if summary.type != "fusion" or not backbones:
        raise ValueError(
            "inference_network.type=grouped_diffusion needs summary_network.type=fusion with "
            "params.backbones -- one droppable group per modality."
        )
    if params.get("head"):
        raise ValueError(
            "grouped_diffusion needs summary_network.params.head to be null: a fusion head mixes "
            "the branch embeddings, so dropping a slice of its output is not dropping a modality."
        )
    names = sorted(backbones.keys())
    sizes = [
        int(OmegaConf.merge(OmegaConf.structured(SummaryNetworkConfig), backbones[k] or {}).summary_dim)
        for k in names
    ]
    return sizes, names


@register_inference_network("grouped_diffusion")
def _grouped_diffusion(cfg, model_cfg):
    """``DiffusionModel`` with per-modality condition dropout; groups read off the fusion config.

    Deriving the group widths from ``summary_network.params.backbones`` rather than restating them
    here keeps a tuning trial that searches ``summary_dim`` correct for free.
    """
    subnet, subnet_kwargs = _subnet_kwargs(cfg)
    sizes, names = _fusion_group_sizes(model_cfg)
    return GroupedDiffusionModel(
        subnet=subnet,
        subnet_kwargs=subnet_kwargs,
        group_sizes=sizes,
        group_names=names,
        missing_modality_prob=float((cfg.params or {}).get("missing_modality_prob", 0.0)),
    )


#: Needs the whole ``cfg.model`` (not just its ``inference_network`` node) to read the summary
#: network's per-backbone widths. See ``registry.build_inference_network``.
_grouped_diffusion.needs_model_cfg = True
