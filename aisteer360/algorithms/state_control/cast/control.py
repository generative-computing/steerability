"""CAST control: conditional activation steering, composed from `common` components."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.state_control.base import InterventionControl
from aisteer360.algorithms.state_control.common.estimators import ContrastiveDirectionEstimator, MeanDifferenceEstimator
from aisteer360.algorithms.state_control.common.fit_specs import Comparator, CompMode, VectorTrainSpec
from aisteer360.algorithms.state_control.common.gating import Gate, PerKeyThreshold
from aisteer360.algorithms.state_control.common.selectors import LateThirdSelector
from aisteer360.algorithms.state_control.common.sources import ConditionPointSearch, _Precomputed
from aisteer360.algorithms.state_control.common.specs import Intervention, TokenScope
from aisteer360.algorithms.state_control.common.transforms import AdditiveTransform, NormPreservingTransform
from aisteer360.algorithms.state_control.common.transforms.base import BaseTransform

from .args import CASTArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionPointConfig:
    """Fully-resolved condition point produced by binding the condition source.

    Attributes:
        layer_ids: Condition layer ids (0-based), empty when unconditional.
        threshold: Gate threshold, or None when unconditional.
        comparator: Gate comparator.
        comparison_mode: Runtime token aggregation for condition scoring ("mean" or "last").
        enabled: Whether conditional gating is active. When False, behavior steering is always on.
    """

    layer_ids: frozenset[int]
    threshold: float | None
    comparator: Comparator
    comparison_mode: CompMode
    enabled: bool


@dataclass(frozen=True)
class CASTDecision:
    """Diagnostics snapshot of the most recent condition decision.

    Attributes:
        scores: Per-condition-layer score for the first logical row (single-prompt convenience).
        scores_per_row: Per-condition-layer scores for every logical row.
        threshold: The gate threshold in effect.
        comparator: The gate comparator in effect.
        open_per_row: Whether the gate opened, per logical row.
    """

    scores: dict[int, float]
    scores_per_row: dict[int, tuple[float, ...]]
    threshold: float | None
    comparator: Comparator | None
    open_per_row: tuple[bool, ...]

    @property
    def is_open(self) -> bool:
        """True if the gate opened for the first logical row (single-prompt convenience)."""
        return bool(self.open_per_row and self.open_per_row[0])


def _make_estimator(spec: VectorTrainSpec):
    """Dispatch a fit spec to its estimator.

    Raises:
        ValueError: If `method == "mean_diff"` is combined with `accumulate == "suffix-only"`,
            which the mean-difference estimator does not support.
    """
    if spec.method == "mean_diff":
        if spec.accumulate == "suffix-only":
            raise ValueError(
                "method='mean_diff' does not support accumulate='suffix-only'; "
                "use accumulate='all' or 'last_token', or method='pca_pairwise'/'pca_center'."
            )
        return MeanDifferenceEstimator()
    return ContrastiveDirectionEstimator()


def _squeeze_direction(d: torch.Tensor) -> torch.Tensor:
    """Squeeze a [1, H] direction to [H] for scalar operations.

    Handles both 1D [H] and 2D [K, D] tensors. For K=1, squeezes to [D].
    For K>1, returns as-is (caller must handle).
    """
    if d.ndim == 2 and d.shape[0] == 1:
        return d.squeeze(0)
    return d


class _BehaviorFit:
    """A fit recipe for CAST's behavior vector, dispatched through `_make_estimator`."""

    access = ModelAccess.MODULE
    artifact_class = "direction"

    def __init__(self, data, fit_spec: VectorTrainSpec):
        self._data = data
        self._fit_spec = fit_spec

    def resolve(self, model, tokenizer, *, session=None):
        estimator = _make_estimator(self._fit_spec)
        return estimator.fit(model, tokenizer, data=self._data, spec=self._fit_spec, session=session)


class _BehaviorBuild:
    """Transform factory for CAST's default additive path.

    Resolves the behavior artifact, squeezes each covered layer's direction, applies the
    explained-variance scaling when enabled, and builds the additive transform (optionally
    norm-preserving). Behavior layers without a fitted direction are skipped, so their hooks
    pass through unchanged. The factory declares its wire plan (`wire_plan`,
    `wire_modifiers`), so kind identity is readable before binding.
    """

    def __init__(self, source, strength: float, use_explained_variance: bool, norm_preserving: bool):
        self._source = source
        self._strength = strength
        self._use_explained_variance = use_explained_variance
        self._norm_preserving = norm_preserving

    @property
    def access(self) -> ModelAccess:
        return getattr(self._source, "access", ModelAccess.MODULE)

    @property
    def artifact_class(self) -> str | None:
        return getattr(self._source, "artifact_class", None)

    def wire_plan(self) -> str | None:
        """`"additive"` for broadcast behavior directions; None when the source is positional."""
        if getattr(self._source, "produces_positional", False):
            return None
        return "additive"

    def wire_modifiers(self) -> tuple[str, ...]:
        """The planned wrapper kinds: `norm_preserving` when OOI normalization is enabled."""
        return ("norm_preserving",) if self._norm_preserving else ()

    def __call__(self, ctx) -> BaseTransform:
        behavior_vec = ctx.resolve(self._source)
        directions: dict[int, torch.Tensor] = {}
        for layer_id in ctx.layer_ids:
            direction = behavior_vec.directions.get(layer_id)
            if direction is None:
                continue
            direction = _squeeze_direction(direction)
            if self._use_explained_variance and behavior_vec.explained_variances:
                direction = direction * float(behavior_vec.explained_variances.get(layer_id, 1.0))
            directions[layer_id] = direction

        base_transform: BaseTransform = AdditiveTransform(directions, strength=self._strength)
        if self._norm_preserving:
            return NormPreservingTransform(base_transform)
        return base_transform


class CAST(InterventionControl):
    """Conditional Activation Steering (CAST).

    CAST enables selective control of LLM behavior by conditionally applying activation steering
    based on input context. It operates in two phases:

    1. **Condition Detection**: Scores hidden-state activation patterns at the condition layer(s)
       against a learned condition direction to detect whether the prompt matches the target
       condition.

    2. **Conditional Behavior Modification**: When the condition is met, applies a behavior
       transform to hidden states at the behavior layers.

    The control is declarative: `_configure` maps the validated args onto one `Intervention`
    at the layer-input boundary whose gate comes from a `ConditionPointSearch` source (fitting
    the condition vector and grid-searching the gate point at bind time), and whose transform
    comes from the default additive build or the `behavior_transform` slot. The runtime pieces
    it resolves to are the `common` component families:

    - `ContrastiveDirectionEstimator` / `MeanDifferenceEstimator`: learn per-layer direction
      vectors from contrastive text pairs.
    - `ConditionPointSelector`: grid-searches the (layer, threshold, comparator) that best
      separates positive from negative calibration examples.
    - `Gate(Evidence(..., ProjectedCosineReadout(...)), PerKeyThreshold(...))`: row-vectorized
      gating. Evidence pooling is pad-aware ("mean" or "last") and each pooled state is scored
      per row via projected cosine similarity. Each prompt in a batch is gated independently;
      beam-expanded rows of one prompt share that prompt's decision; the decision freezes after
      the prefill pass (the runtime stops condition scoring once the gate reports ready).
    - The behavior transform: `AdditiveTransform` (scaled direction addition, optionally wrapped
      in `NormPreservingTransform`) by default, or any `BaseTransform` supplied via
      `behavior_transform` (e.g. `ProjectionTransform` for conditional ablation).

    The intervention applies at the layer-input boundary. Behavior directions are estimated at
    the output of layer l (`hidden_states[l+1]`) and applied at the input of layer l (the
    output of layer l-1), a one-layer skew. Condition directions are estimated by default at
    the input of layer l (`VectorTrainSpec(location="layer_input")` in `CASTArgs.condition_fit`),
    the boundary the `ConditionPointSelector` calibrates on and the runtime condition pre-hook
    scores, so condition fit, calibration, and runtime are aligned.

    Within the prefill pass, hooks fire in layer order. A behavior layer below the condition layer
    sees a still-closed gate with no evidence yet, while a behavior layer above it sees the decided
    gate. When the calibrated condition layer sits above the behavior layers, prompt tokens pass
    the behavior layers unsteered and only decode steps are steered. Token scope composes with
    this: `"all"` permits prompt-token steering wherever the gate is already decided during
    prefill; `"after_prompt"` restricts steering to generated tokens regardless of layer order.

    Batching is supported (`supports_batching = True`). Row-vectorized gates let one batched
    `generate` call gate and steer each prompt independently.

    Reference:

    - "Programming Refusal with Conditional Activation Steering"
    Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin,
    Manish Nagireddy, Amit Dhurandhar
    [https://arxiv.org/abs/2409.05907](https://arxiv.org/abs/2409.05907)
    """

    Args = CASTArgs
    supports_batching = True

    def _configure(self):
        if self.behavior_transform is not None:
            transform = self.behavior_transform
            require_coverage = True
        else:
            if self.behavior_vector is not None:
                source = _Precomputed(self.behavior_vector.clone())
            else:
                source = _BehaviorFit(self.behavior_data, self.behavior_fit)
            transform = _BehaviorBuild(
                source,
                strength=self.behavior_vector_strength,
                use_explained_variance=self.use_explained_variance,
                norm_preserving=self.use_ooi_preventive_normalization,
            )
            require_coverage = False

        self._condition_source = ConditionPointSearch(
            condition_vector=self.condition_vector.clone() if self.condition_vector is not None else None,
            condition_data=self.condition_data,
            condition_fit=self.condition_fit,
            search=self.search,
            layer_ids=self.condition_layer_ids,
            threshold=self.condition_vector_threshold,
            comparator=self.condition_comparator_threshold_is,
            comparison_mode=self.condition_threshold_comparison_mode,
        )

        self._template = (Intervention(
            layers=tuple(sorted(set(int(lid) for lid in self.behavior_layer_ids)))
                   if self.behavior_layer_ids is not None else LateThirdSelector(),
            transform=transform,
            scope=TokenScope(self.token_scope, last_k=self.last_k, from_position=self.from_position),
            gate=self._condition_source,
            boundary="layer_input",
            require_coverage=require_coverage,
        ),)

    @property
    def _behavior_layer_ids(self) -> list[int]:
        """The resolved behavior layers (empty before `steer()`)."""
        return list(self.interventions[0].layers) if self.interventions else []

    @property
    def _threshold_rule(self) -> PerKeyThreshold | None:
        """The gate's threshold rule, for diagnostics; None when unconditional or unbound."""
        gate = self._gate
        if isinstance(gate, Gate) and isinstance(gate.rule, PerKeyThreshold):
            return gate.rule
        return None

    @property
    def _cond_config(self) -> ConditionPointConfig | None:
        """The resolved condition point, or None before `steer()`."""
        if not self.interventions:
            return None
        point = self._condition_source.resolved_point
        if point is None:
            return ConditionPointConfig(
                layer_ids=frozenset(),
                threshold=None,
                comparator=self.condition_comparator_threshold_is,
                comparison_mode=self.condition_threshold_comparison_mode,
                enabled=False,
            )
        return ConditionPointConfig(
            layer_ids=frozenset(point["layer_ids"]),
            threshold=point["threshold"],
            comparator=point["comparator"],
            comparison_mode=point["comparison_mode"],
            enabled=True,
        )

    @property
    def latest_decision(self) -> CASTDecision | None:
        """The most recent condition decision, or None before the condition has been evaluated.

        Assembled on demand from the gate's retained evidence; cleared when the next
        generation's hooks are built.
        """
        rule = self._threshold_rule
        gate = self._gate
        if rule is None or gate is None or not gate.is_ready():
            return None
        evidence = gate.evidence_values()
        if not evidence:
            return None
        open_rows = gate.open_rows()
        return CASTDecision(
            scores={lid: float(rows[0]) for lid, rows in evidence.items()},
            scores_per_row={lid: tuple(float(x) for x in rows) for lid, rows in evidence.items()},
            threshold=rule.threshold,
            comparator=rule.comparator,
            open_per_row=tuple(bool(x) for x in open_rows.tolist()),
        )

    @property
    def condition_point(self) -> dict | None:
        """The resolved condition configuration, or None when the control is unconditional.

        Populated by `steer()` from either the auto-search or the caller-supplied
        `condition_layer_ids` / `condition_vector_threshold` / `condition_comparator_threshold_is`.
        """
        cfg = self._cond_config
        if cfg is None or not cfg.enabled:
            return None
        return {
            "layer_ids": sorted(cfg.layer_ids),
            "threshold": cfg.threshold,
            "comparator": cfg.comparator,
            "comparison_mode": cfg.comparison_mode,
        }

    def cleanup(self) -> None:
        """Drop references to fitted artifacts and runtime state."""
        self.interventions = ()
