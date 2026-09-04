"""Angular Steering argument validation."""
import math
from dataclasses import dataclass, field
from typing import Literal

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.internals.data import ContrastivePairs, as_contrastive_pairs
from aisteer360.algorithms.state_control.common.fit_specs import VectorTrainSpec
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.token_scope import ScopeKind


@dataclass
class AngularSteeringArgs(BaseArgs):
    """Arguments for Angular Steering.

    Users provide EITHER a pre-computed steering vector (whose per-layer directions must be a
    `[2, H]` orthonormal-ish basis pair) OR contrastive training data (from which the plane is
    fitted during `steer()` via `SteeringPlaneEstimator`).

    The rotation angle may be given in degrees (`target_degree`, the paper's convention) or in
    radians (`angle`); exactly one is allowed and both resolve to `angle_radians`.

    Attributes:
        steering_vector: Pre-computed `[2, H]`-per-layer plane. If provided, skip fitting.
        data: Contrastive pairs for fitting the plane. Required if `steering_vector` is None.
        train_spec: Controls extraction method and accumulation mode for the feature axis.
        target_degree: Rotation angle in degrees. Mutually exclusive with `angle`.
        angle: Rotation angle in radians. Mutually exclusive with `target_degree`.
        mode: `"target"` rotates to the absolute angle (paper Eq. 2); `"offset"` rotates by the
            angle (recovers vector-addition / ablation as special cases).
        adaptive: If True, wrap the rotation in `AlignmentAdaptiveTransform` so only tokens
            aligned with the feature axis are rotated (paper Eq. 3).
        adaptive_threshold: Alignment cutoff for the adaptive gate.
        adaptive_use_cosine: If True, the adaptive gate uses cosine similarity rather than the
            raw projection.
        layer_range: Half-open `[start, end)` range of layers to steer. If None, steer every
            layer for which a plane exists.
        intervention_point: Where the rotation applies. `"norms"` (default) pre-hooks each
            active layer's two normalization sub-modules, rotating the stream entering the
            layer and the mid-layer stream after attention. `"layer_output"` forward-hooks each
            active decoder layer, rotating its residual-stream output once per layer; this is
            the only placement with an intervention-spec form, since the mid-layer boundary
            exists only inside the in-process forward pass.
        use_norm_preservation: If True, additionally wrap in `NormPreservingTransform` as a guard
            against float drift (rotation already preserves norm by construction).
        token_scope: Which tokens to steer (see `make_token_mask`).
        last_k: Required when `token_scope == "last_k"`.
        from_position: Required when `token_scope == "from_position"`.
        angle_radians: Resolved rotation angle in radians (computed in `__post_init__`).
    """

    # steering plane source (provide exactly one)
    steering_vector: SteeringVector | None = None
    data: ContrastivePairs | dict | None = None

    # training configuration
    train_spec: VectorTrainSpec | dict = field(
        default_factory=lambda: VectorTrainSpec(method="mean_diff", accumulate="last_token")
    )

    # rotation configuration
    target_degree: float | None = None  # degrees, XOR angle
    angle: float | None = None  # radians, XOR target_degree
    mode: Literal["target", "offset"] = "target"

    # adaptive gating
    adaptive: bool = False
    adaptive_threshold: float = 0.0
    adaptive_use_cosine: bool = False

    # layer / norm configuration
    layer_range: tuple[int, int] | None = None  # half-open [start, end)
    intervention_point: Literal["norms", "layer_output"] = "norms"
    use_norm_preservation: bool = False

    # inference configuration
    token_scope: ScopeKind = "all"
    last_k: int | None = None
    from_position: int | None = None

    # resolved in __post_init__
    angle_radians: float = field(init=False, default=0.0)

    def __post_init__(self):
        # exactly one of steering_vector or data must be provided
        if self.steering_vector is None and self.data is None:
            raise ValueError("Provide either steering_vector or data.")
        if self.steering_vector is not None and self.data is not None:
            raise ValueError("Provide steering_vector or data, not both.")

        # validate a precomputed vector and require the K=2 basis-pair shape
        if self.steering_vector is not None:
            self.steering_vector.validate()
            for layer_id, dirs in self.steering_vector.directions.items():
                if dirs.ndim != 2 or dirs.size(0) != 2:
                    raise ValueError(
                        f"Angular steering requires [2, H] directions (basis pair); "
                        f"got shape {tuple(dirs.shape)} at layer {layer_id}."
                    )

        # normalize dict inputs
        if self.data is not None and not isinstance(self.data, ContrastivePairs):
            object.__setattr__(self, "data", as_contrastive_pairs(self.data))

        if isinstance(self.train_spec, dict):
            object.__setattr__(self, "train_spec", VectorTrainSpec(**self.train_spec))

        # resolve rotation angle (degrees XOR radians)
        if self.target_degree is not None and self.angle is not None:
            raise ValueError("Provide target_degree or angle, not both.")
        if self.target_degree is not None:
            object.__setattr__(self, "angle_radians", math.radians(float(self.target_degree)))
        elif self.angle is not None:
            object.__setattr__(self, "angle_radians", float(self.angle))
        else:
            object.__setattr__(self, "angle_radians", 0.0)

        # validate mode
        if self.mode not in ("target", "offset"):
            raise ValueError(f"mode must be 'target' or 'offset'; got {self.mode!r}.")

        # validate intervention point
        if self.intervention_point not in ("norms", "layer_output"):
            raise ValueError(
                f"intervention_point must be 'norms' or 'layer_output'; got {self.intervention_point!r}."
            )

        # validate layer_range
        if self.layer_range is not None:
            start, end = self.layer_range
            if start < 0 or end <= start:
                raise ValueError(f"layer_range must be a half-open [start, end) with 0 <= start < end; got {self.layer_range}.")

        # token scope cross-checks
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
        if self.token_scope == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when token_scope is 'from_position'.")
