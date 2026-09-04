"""CAST argument validation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.internals.data import ContrastivePairs, as_contrastive_pairs
from aisteer360.algorithms.state_control.common.fit_specs import (
    Comparator,
    CompMode,
    ConditionSearchSpec,
    VectorTrainSpec,
)
from aisteer360.algorithms.state_control.common.selectors.condition_point import ConditionPoint
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.token_scope import ScopeKind
from aisteer360.algorithms.state_control.common.transforms.base import BaseTransform

if TYPE_CHECKING:
    from aisteer360.algorithms.state_control.common.transforms.context import TransformContext


@dataclass
class CASTArgs(BaseArgs):
    """Arguments for CAST (Conditional Activation Steering).

    The applied behavior artifact comes from exactly one of three routes: a pre-computed
    `behavior_vector`, contrastive `behavior_data` fitted during steer(), or a `behavior_transform`
    that carries its own artifact. The first two feed the default additive path (fitted vectors are
    scaled by `behavior_vector_strength` and added); `behavior_transform` replaces that construction
    with any `BaseTransform`.

    All layer validation happens in steer() once the model is known.

    Attributes:
        behavior_vector: Pre-computed behavior steering vector.
        behavior_data: Contrastive pairs for training the behavior vector.
        behavior_fit: Training configuration for behavior vector extraction. Applies to the default
            additive path; invalid alongside `behavior_transform`, which forces `behavior_data` to be
            absent and so never fits (a source-carrying transform configures its own fit via
            `ContrastiveFit`).
        behavior_layer_ids: Layers to apply the behavior artifact to. If None,
            defaults to the late third of the model's layers.
        behavior_transform: An alternative behavior application, replacing the default additive
            construction. Accepts a `BaseTransform` (bound, e.g.
            `ProjectionTransform(vector, alpha=0.8)`, or source-carrying, e.g.
            `ProjectionTransform(ContrastiveFit(data=...))` bound at steer()), or a factory
            `Callable[[TransformContext], BaseTransform]`. The transform is the sole artifact carrier,
            so it is mutually exclusive with `behavior_vector`/`behavior_data` and with the additive
            knobs `behavior_vector_strength`, `use_explained_variance`, and
            `use_ooi_preventive_normalization` (incorporate those into the transform instead). The transform
            is applied at the pre-hook input of the behavior layers, the same one-layer skew as the
            additive path; a `ContrastiveFit` source defaults to `location="layer_output"` to match
            the behavior convention, and users who want the artifact fit at the applied boundary pass
            `ContrastiveFit(..., location="layer_input")`.
        behavior_vector_strength: Scaling factor for the behavior vector.
            Positive values induce the target behavior; negative values subtract
            it (e.g., negate a refusal vector to remove refusal), matching the
            sign convention of the paper's behavior strength. Applies to the default additive path;
            invalid alongside `behavior_transform`.
        condition_vector: Pre-computed condition steering vector.
        condition_data: Contrastive pairs for training the condition vector.
        condition_fit: Training configuration for condition vector extraction. Defaults to fitting at
            the layer-input boundary (`location="layer_input"`), matching both the
            `ConditionPointSelector` calibration and the runtime condition pre-hook.
        search: Configuration for automatic condition point search.
        condition_point: A complete, reusable condition point supplied as a single object. Accepts a
            `ConditionPoint` (e.g. from a prior `ConditionPointSelector` search, invertible via
            `.flipped()`) or the dict returned by the `CAST.condition_point` property (keys
            `layer_ids`, `threshold`, `comparator`, and optionally `comparison_mode`). Expanded in
            `__post_init__` into `condition_layer_ids` / `condition_vector_threshold` /
            `condition_comparator_threshold_is` (and `condition_threshold_comparison_mode` when the
            point carries one). This is a complete manual configuration and supersedes
            `search.auto_find`; it is mutually exclusive with `condition_layer_ids` and
            `condition_vector_threshold`.
        condition_layer_ids: Layers to check the condition on.
        condition_vector_threshold: Similarity threshold for condition detection.
        condition_comparator_threshold_is: When to open the gate. `"ge"` opens when
            score >= threshold and `"le"` opens when score <= threshold.
        condition_threshold_comparison_mode: How to aggregate hidden states
            for comparison ("mean" or "last").
        use_ooi_preventive_normalization: Apply out-of-distribution preventive
            normalization to maintain hidden state magnitudes. Applies to the default additive path;
            invalid alongside `behavior_transform` (wrap the transform in `NormPreservingTransform`
            instead).
        use_explained_variance: Scale steering vectors by their explained
            variance for adaptive layer-wise control. Only the PCA methods
            ("pca_center", "pca_pairwise") produce explained variances; with
            method="mean_diff" there is no variance to scale by and this is a
            no-op. Applies to the default additive path; invalid alongside `behavior_transform`
            (pre-scale the artifact instead).
        token_scope: Which tokens to steer ("all", "after_prompt", "last_k", "from_position").
            `"all"` (default, reference-faithful) permits prompt-token steering during prefill
            wherever the gate is already decided (i.e., at behavior layers above the condition
            layer) and steers every decode token. `"after_prompt"` steers only generated tokens.
            This subsumes the former `apply_behavior_on_first_call` flag:
            `apply_behavior_on_first_call=True` corresponds to `token_scope="all"` and
            `apply_behavior_on_first_call=False` to `token_scope="after_prompt"`.
        last_k: Required when token_scope == "last_k".
        from_position: Required when token_scope == "from_position". The absolute position from
            which to start steering (for single-pass logit scoring).
    """

    # behavior
    behavior_vector: SteeringVector | None = None
    behavior_data: ContrastivePairs | dict | None = None
    behavior_fit: VectorTrainSpec = field(
        default_factory=lambda: VectorTrainSpec(method="pca_center", prompt_format="raw")
    )
    behavior_layer_ids: Sequence[int] | None = None
    behavior_transform: BaseTransform | Callable[["TransformContext"], BaseTransform] | None = None
    behavior_vector_strength: float = 1.0

    # condition
    condition_vector: SteeringVector | None = None
    condition_data: ContrastivePairs | dict | None = None
    condition_fit: VectorTrainSpec = field(
        default_factory=lambda: VectorTrainSpec(
            method="pca_center", accumulate="all", prompt_format="chat_prompt",
            location="layer_input",
        )
    )
    search: ConditionSearchSpec = field(default_factory=ConditionSearchSpec)
    condition_point: ConditionPoint | Mapping | None = None
    condition_layer_ids: Sequence[int] | None = None
    condition_vector_threshold: float | None = None
    condition_comparator_threshold_is: Comparator = "ge"
    condition_threshold_comparison_mode: CompMode = "mean"

    # hook behavior
    use_ooi_preventive_normalization: bool = False
    use_explained_variance: bool = False
    token_scope: ScopeKind = "all"
    last_k: int | None = None
    from_position: int | None = None

    def __post_init__(self):
        if self.behavior_vector is not None:
            self.behavior_vector.validate()
        if self.condition_vector is not None:
            self.condition_vector.validate()

        if self.condition_comparator_threshold_is not in ("ge", "le"):
            raise ValueError(
                f"condition_comparator_threshold_is must be 'ge' or 'le'; "
                f"got {self.condition_comparator_threshold_is!r}."
            )

        # expand a reusable condition point into the manual triple (supersedes search.auto_find)
        if self.condition_point is not None:
            if self.condition_layer_ids is not None or self.condition_vector_threshold is not None:
                raise ValueError(
                    "condition_point already carries the layers and threshold; drop "
                    "condition_layer_ids and condition_vector_threshold, or drop condition_point."
                )

            if isinstance(self.condition_point, ConditionPoint):
                point_layer_ids: Sequence[int] = [self.condition_point.layer_id]
                point_threshold = self.condition_point.threshold
                point_comparator = self.condition_point.comparator
                point_comparison_mode = self.condition_point.comparison_mode
            elif isinstance(self.condition_point, Mapping):
                try:
                    point_layer_ids = list(self.condition_point["layer_ids"])
                    point_threshold = self.condition_point["threshold"]
                    point_comparator = self.condition_point["comparator"]
                except KeyError as exc:
                    raise ValueError(
                        "condition_point dict must have keys 'layer_ids', 'threshold', and "
                        f"'comparator' (missing {exc.args[0]!r})."
                    ) from None
                point_comparison_mode = self.condition_point.get("comparison_mode")
            else:
                raise TypeError(
                    "condition_point must be a ConditionPoint or a mapping with keys 'layer_ids', "
                    f"'threshold', 'comparator'; got {type(self.condition_point).__name__}."
                )

            if len(point_layer_ids) == 0:
                raise ValueError(
                    "condition_point carries no condition layers; a condition point must name at "
                    "least one layer (a complete manual configuration must not degrade to "
                    "unconditional steering)."
                )
            if point_comparison_mode is not None and point_comparison_mode not in ("mean", "last"):
                raise ValueError(
                    f"condition_point comparison_mode must be 'mean' or 'last', got "
                    f"{point_comparison_mode!r}."
                )

            if point_comparator not in ("ge", "le"):
                raise ValueError(
                    f"condition_point comparator must be 'ge' or 'le'; got {point_comparator!r}."
                )
            self.condition_layer_ids = list(point_layer_ids)
            self.condition_vector_threshold = point_threshold
            self.condition_comparator_threshold_is = point_comparator
            if point_comparison_mode is not None:
                self.condition_threshold_comparison_mode = point_comparison_mode

        # normalize dict inputs to ContrastivePairs
        if self.behavior_data is not None and not isinstance(self.behavior_data, ContrastivePairs):
            object.__setattr__(self, "behavior_data", as_contrastive_pairs(self.behavior_data))
        if self.condition_data is not None and not isinstance(self.condition_data, ContrastivePairs):
            object.__setattr__(self, "condition_data", as_contrastive_pairs(self.condition_data))

        # behavior artifact source: exactly one of vector / data / transform
        if self.behavior_transform is not None:
            if not (isinstance(self.behavior_transform, BaseTransform) or callable(self.behavior_transform)):
                raise TypeError(
                    "behavior_transform must be a BaseTransform or a factory "
                    "Callable[[TransformContext], BaseTransform]; got "
                    f"{type(self.behavior_transform).__name__}."
                )
            if self.behavior_vector is not None or self.behavior_data is not None:
                raise ValueError(
                    "behavior_transform carries its own artifact; provide the vector, data, or source "
                    "on the transform (e.g. ProjectionTransform(ContrastiveFit(data=...))) "
                    "instead of behavior_vector/behavior_data."
                )
            if self.behavior_vector_strength != 1.0:
                raise ValueError(
                    "behavior_vector_strength scales the default additive path and has no referent "
                    "with behavior_transform; construct the transform with its own magnitude "
                    "(e.g. AdditiveTransform(..., strength=...) or ProjectionTransform(..., "
                    "alpha=...))."
                )
            if self.use_ooi_preventive_normalization:
                raise ValueError(
                    "use_ooi_preventive_normalization wraps the default additive path and does not "
                    "apply with behavior_transform; wrap the transform instead "
                    "(NormPreservingTransform(inner))."
                )
            if self.use_explained_variance:
                raise ValueError(
                    "use_explained_variance scales the default additive path and has no "
                    "behavior_transform equivalent; pre-scale the artifact the transform carries."
                )
        elif self.behavior_vector is None and self.behavior_data is None:
            raise ValueError("Provide behavior_vector, behavior_data, or behavior_transform.")

        # if condition vector is given, condition layers should also be given
        # (or search.auto_find should be True)
        if self.condition_vector is not None and self.condition_layer_ids is None and not self.search.auto_find:
            raise ValueError(
                "When condition_vector is provided without condition_layer_ids, "
                "search.auto_find must be True."
            )

        # a partial manual condition point must fail rather than degrade to unconditional steering
        if not self.search.auto_find:
            if self.condition_layer_ids is not None and self.condition_vector_threshold is None:
                raise ValueError(
                    "condition_layer_ids given without condition_vector_threshold; provide a "
                    "threshold or enable search.auto_find."
                )
            if self.condition_vector_threshold is not None and self.condition_layer_ids is None:
                raise ValueError(
                    "condition_vector_threshold given without condition_layer_ids; provide "
                    "condition layers or enable search.auto_find."
                )
            if self.condition_data is not None and self.condition_layer_ids is None:
                raise ValueError(
                    "condition_data given with search.auto_find=False and no manual condition point; "
                    "enable search.auto_find or provide condition_layer_ids and a threshold."
                )

        # token scope cross-checks
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
        if self.token_scope == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when token_scope is 'from_position'.")
