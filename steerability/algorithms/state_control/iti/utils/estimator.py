"""Probe-based mass mean shift estimator for ITI.

The per-head classifiers this estimator fits are fit-time selection statistics. Their held-out
accuracies rank heads and their weights are discarded. They are ordinary scikit-learn logistic
regressions over per-head activation slices, unrelated to the toolkit's `Probe` detector (a
calibrated, model-free readout over pooled residual-stream features consumed by gates and
routers). "Probe" here means only ITI's fit-time head classifier.
"""
import logging
import warnings
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import OptimizeWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.internals.data import LabeledExamples
from steerability.algorithms.core.internals.encoding import tokenize_texts
from steerability.algorithms.core.internals.fingerprint import artifact_provenance_meta
from steerability.algorithms.core.internals.model_layout import head_geometry, resolve_model_layout
from steerability.algorithms.core.internals.pooling import get_last_token_positions, masked_mean, select_at_positions
from steerability.algorithms.state_control.common.estimators.base import BaseEstimator
from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec
from steerability.algorithms.state_control.common.steering_vector import SteeringVector

logger = logging.getLogger(__name__)

_VAL_FRACTION = 0.2
_SPLIT_SEED = 42


class _MaskHolder:
    """Carries the current chunk's attention mask so capture hooks can pool per row."""

    __slots__ = ("mask",)

    def __init__(self):
        self.mask: torch.Tensor | None = None


class ProbeMassShiftEstimator(BaseEstimator[SteeringVector]):
    """Learns per-head direction vectors using probe-based mass mean shift.

    The fit has three parts, run per attention head across every layer:

    1. Extract the head's pre-`o_proj` activation on labeled positive/negative statements, pooled
       to one vector per statement.
    2. Fit a logistic-regression probe on an 80/20 held-out split and record its validation
       accuracy for later head selection. The split is drawn over groups when the data carries
       group keys, so no group's statements straddle the train/validation partition.
    3. Compute the mass-mean-shift direction as `mean(positives) - mean(negatives)`, L2-normalize
       it to a unit direction, and scale it by the standard deviation of all statements projected
       onto that direction.

    Returns a `SteeringVector` with directions shaped `[num_heads, head_dim]` per layer,
    `num_heads`/`head_dim` metadata, and `probe_accuracies` populated for every `(layer, head)`
    pair. "Probe" here is the fit-time head classifier used to rank heads, not the toolkit's
    `Probe` detector.

    Per-token activations are pooled inside the capture hook and never retained beyond the chunk
    that produced them, so host memory scales with the number of statements rather than their
    length.

    Reference:

        - "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
          Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg
          [https://arxiv.org/abs/2306.03341](https://arxiv.org/abs/2306.03341)
    """

    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: LabeledExamples,
        spec: VectorTrainSpec,
    ) -> SteeringVector:
        """Extract per-head steering directions and probe accuracies from labeled statements.

        Args:
            model: Model to extract attention outputs from.
            tokenizer: Tokenizer for encoding the labeled examples.
            data: Independent positive/negative statements (true/false statements for ITI). Unlike
                `ContrastivePairs`, these do not need to be equal length. When the data carries
                group keys, the probe train/validation split is drawn over groups.
            spec: Training configuration (`accumulate` mode and `batch_size`).

        Returns:
            A `SteeringVector` with directions shaped `[num_heads, head_dim]` per layer,
            `num_heads`/`head_dim` metadata, `probe_accuracies` for every head, and
            `meta["probe_split"]` recording whether the split was drawn over groups
            (`"group"`) or statements (`"statement"`).

        Raises:
            ValueError: If the model is a hybrid attention stack (some decoder layers carry no
                attention module), its attention head geometry is not uniform across layers,
                `spec.accumulate` is unsupported, or the probe partition cannot be drawn.
        """
        model_type = getattr(model.config, "model_type", "unknown")

        # ITI reshapes every layer's o_proj input with one head geometry, so every layer must carry
        # an attention module and the geometry must be uniform. refuse hybrid stacks, then read the
        # geometry per layer from the module tree and fail loudly before any capture on a model with
        # heterogeneous heads (e.g. Gemma 4 alternates sliding and global head dims).
        layout = resolve_model_layout(model)
        if layout.is_hybrid:
            raise ValueError(
                "ITI requires an attention module on every decoder layer, but only layers "
                f"{list(layout.attention_layers)} of {layout.num_layers} carry one; hybrid "
                "attention stacks such as Qwen3.5 are not supported by ITI."
            )
        geometries = {lid: head_geometry(model, layout, lid) for lid in range(layout.num_layers)}
        distinct = {(g.num_heads, g.head_dim) for g in geometries.values()}
        if len(distinct) > 1:
            differing = sorted((lid, g.num_heads, g.head_dim) for lid, g in geometries.items())
            raise ValueError(
                "ITI requires uniform attention head geometry across layers, but the model has "
                f"heterogeneous heads: {differing} as (layer, num_heads, head_dim). Models such as "
                "Gemma 4 that alternate sliding and global head dimensions are not supported by ITI."
            )
        geometry = next(iter(geometries.values()))
        num_heads = geometry.num_heads
        head_dim = geometry.head_dim

        if spec.accumulate not in ("last_token", "all"):
            raise ValueError(f"ProbeMassShiftEstimator does not support accumulate='{spec.accumulate}'.")

        device = next(model.parameters()).device
        pos_texts = list(data.positives)
        neg_texts = list(data.negatives)
        n_pos = len(pos_texts)
        n_neg = len(neg_texts)

        logger.debug("Tokenizing %d positive and %d negative statements", n_pos, n_neg)
        enc_pos = tokenize_texts(tokenizer, pos_texts, device)
        enc_neg = tokenize_texts(tokenizer, neg_texts, device)

        logger.debug("Extracting pooled head features with batch_size=%d", spec.batch_size)
        feats_pos = _extract_pooled_head_features(model, enc_pos, spec.batch_size, spec.accumulate)
        feats_neg = _extract_pooled_head_features(model, enc_neg, spec.batch_size, spec.accumulate)

        labels = np.array([1] * n_pos + [0] * n_neg)
        groups = None
        if data.groups:
            groups = list(data.positive_groups) + list(data.negative_groups)
        train_idx, val_idx = _probe_partition(labels, groups)

        num_layers = len(feats_pos)
        logger.debug("Computing probe-based directions for %d layers x %d heads", num_layers, num_heads)

        directions: dict[int, torch.Tensor] = {}
        probe_accuracies: dict[tuple[int, int], float] = {}

        # sklearn's lbfgs fit warns under recent scipy (iprint) and at the iteration cap; neither affects the fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OptimizeWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            for layer_id in range(num_layers):
                pos_heads = feats_pos[layer_id].view(n_pos, num_heads, head_dim)
                neg_heads = feats_neg[layer_id].view(n_neg, num_heads, head_dim)

                layer_directions = []
                for head_id in range(num_heads):
                    hp = pos_heads[:, head_id, :].float()  # [n_pos, head_dim]
                    hn = neg_heads[:, head_id, :].float()  # [n_neg, head_dim]
                    x = torch.cat([hp, hn], dim=0)  # [N, head_dim]
                    features = x.numpy()

                    probe = LogisticRegression(max_iter=1000, solver="lbfgs")
                    probe.fit(features[train_idx], labels[train_idx])
                    probe_accuracies[(layer_id, head_id)] = float(probe.score(features[val_idx], labels[val_idx]))

                    raw = hp.mean(dim=0) - hn.mean(dim=0)  # [head_dim]
                    norm = raw.norm()
                    theta_hat = raw / norm if norm > 0 else raw
                    sigma = (x @ theta_hat).std()
                    layer_directions.append((sigma * theta_hat).to(dtype=torch.float32))

                directions[layer_id] = torch.stack(layer_directions, dim=0)  # [num_heads, head_dim]

        logger.debug("Finished fitting probe-based head directions")
        meta = {
            **artifact_provenance_meta(model, tokenizer),
            "probe_split": "group" if groups is not None else "statement",
        }
        return SteeringVector(
            model_type=model_type,
            directions=directions,
            num_heads=num_heads,
            head_dim=head_dim,
            probe_accuracies=probe_accuracies,
            meta=meta,
        )


@torch.no_grad()
def _extract_pooled_head_features(
    model: PreTrainedModel,
    enc: dict[str, torch.Tensor],
    batch_size: int,
    accumulate: str,
) -> dict[int, torch.Tensor]:
    """Extract pooled pre-`o_proj` head features from every layer via temporary hooks.

    Registers one `forward_pre_hook` on each layer's output projection (`o_proj` / `c_proj`),
    which captures the concatenated per-head attention output before the projection mixes it. Each
    chunk's `[b, T, D]` input is pooled to `[b, D]` on the model's device inside the hook, then
    moved to CPU in the captured dtype and appended to the layer's list. Nothing 3-D is ever
    retained, so host memory scales with the number of statements rather than their length.

    Args:
        model: The model to extract from.
        enc: Tokenized input with `input_ids` and (optionally) `attention_mask`.
        batch_size: Chunk size for the forward passes.
        accumulate: `"last_token"` selects the last non-pad position per row; `"all"` mean-pools
            over the non-pad positions.

    Returns:
        A mapping from `layer_id` to a `[N, num_heads * head_dim]` CPU tensor in the captured dtype.
    """
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    num_examples = input_ids.size(0)

    layout = resolve_model_layout(model)
    oproj_names = layout.oproj_names
    num_layers = layout.num_layers

    storage: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    holder = _MaskHolder()
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_pre_hook(layer_id: int):
        def hook(_module, args, kwargs):
            x = args[0] if args else kwargs.get("input")  # [b, T, D]
            mask = holder.mask
            batch = x.size(0)
            if accumulate == "last_token":
                positions = get_last_token_positions(mask, x.size(1), batch)
                pooled = select_at_positions(x, positions.to(x.device))  # [b, D]
            else:
                pooled = masked_mean(x, mask)  # [b, D]
            storage[layer_id].append(pooled.detach().to("cpu"))
        return hook

    try:
        for layer_id, oproj_name in enumerate(oproj_names):
            oproj_module = model.get_submodule(oproj_name)
            handles.append(oproj_module.register_forward_pre_hook(make_pre_hook(layer_id), with_kwargs=True))

        for start in range(0, num_examples, batch_size):
            end = min(start + batch_size, num_examples)
            batch_ids = input_ids[start:end]
            batch_mask = attention_mask[start:end] if attention_mask is not None else None
            holder.mask = batch_mask
            model(input_ids=batch_ids, attention_mask=batch_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    return {layer_id: torch.cat(tensors, dim=0) for layer_id, tensors in storage.items()}


def _probe_partition(
    labels: np.ndarray,
    groups: Sequence[str | int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one train/validation partition over the concatenated positives-then-negatives space.

    One partition is computed before the head loop and shared by every head. Without groups the
    split is stratified by label; with groups it is drawn over groups so that no group's statements
    appear on both sides.

    Args:
        labels: Binary labels over the concatenated index space (positives then negatives).
        groups: Group key per statement in the same index space, or None for a stratified split.

    Returns:
        A `(train_idx, val_idx)` pair of integer index arrays.

    Raises:
        ValueError: If a grouped partition has fewer than two distinct groups, or either side of
            the resulting partition lacks a class.
    """
    indices = np.arange(len(labels))
    if groups is None:
        train_idx, val_idx = train_test_split(
            indices, test_size=_VAL_FRACTION, random_state=_SPLIT_SEED, stratify=labels
        )
        return train_idx, val_idx

    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        raise ValueError(
            "Grouped probe split requires at least two distinct groups; add data or regroup."
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=_VAL_FRACTION, random_state=_SPLIT_SEED)
    train_idx, val_idx = next(splitter.split(indices, labels, groups))
    if len(np.unique(labels[train_idx])) < 2 or len(np.unique(labels[val_idx])) < 2:
        raise ValueError(
            "Grouped probe split left one side without both classes; add data or regroup."
        )
    return train_idx, val_idx
