"""CAA argument validation."""
from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.internals.data import ContrastivePairs, as_contrastive_pairs
from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec
from steerability.algorithms.state_control.common.sources import ArtifactSource
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.token_scope import ScopeKind


@dataclass
class CAAArgs(BaseArgs):
    """Arguments for CAA (Contrastive Activation Addition).

    Users provide EITHER a pre-computed steering vector OR training data.
    If data is provided, the vector is fitted during steer().

    Attributes:
        steering_vector: Pre-trained steering vector, or an `ArtifactSource` resolving to one
            at `steer()` time (e.g. a provenance-checked precomputed source). If provided,
            skip training. `normalize_vector` requires a concrete vector.
        data: Contrastive pairs for training. Required if steering_vector is None.
        train_spec: Controls extraction method and accumulation mode.
        layer_id: Single layer to apply steering at. If None, uses heuristic.
        multiplier: Scaling factor for the steering vector. Positive increases
            the target behavior, negative decreases it.
        token_scope: Which tokens to steer. "after_prompt" steers only generated
            tokens (for generation). "from_position" steers from a specific position
            within the prompt (for single-pass logit scoring). "all" steers all tokens.
        last_k: Required when token_scope == "last_k".
        from_position: Required when token_scope == "from_position". The absolute
            position from which to start steering.
        normalize_vector: If True, L2-normalize the steering vector before applying.
        use_norm_preservation: If True, wrap transform in NormPreservingTransform.
    """

    # steering vector source (provide exactly one)
    steering_vector: "SteeringVector | ArtifactSource | None" = None
    data: ContrastivePairs | dict | None = None

    # training configuration
    train_spec: VectorTrainSpec | dict = field(
        default_factory=lambda: VectorTrainSpec(method="mean_diff", accumulate="last_token")
    )

    # inference configuration
    layer_id: int | None = None
    multiplier: float = 1.0
    token_scope: ScopeKind = "after_prompt"
    last_k: int | None = None
    from_position: int | None = None
    normalize_vector: bool = False
    use_norm_preservation: bool = False

    def __post_init__(self):
        # exactly one of steering_vector or data must be provided
        if self.steering_vector is None and self.data is None:
            raise ValueError("Provide either steering_vector or data.")
        if self.steering_vector is not None and self.data is not None:
            raise ValueError("Provide steering_vector or data, not both.")

        # validate steering_vector if provided as a concrete vector; sources resolve at steer()
        if isinstance(self.steering_vector, SteeringVector):
            self.steering_vector.validate()
        elif self.steering_vector is not None and self.normalize_vector:
            raise ValueError("normalize_vector requires a concrete steering_vector or data.")

        # normalize dict inputs
        if self.data is not None and not isinstance(self.data, ContrastivePairs):
            object.__setattr__(self, "data", as_contrastive_pairs(self.data))

        if isinstance(self.train_spec, dict):
            object.__setattr__(self, "train_spec", VectorTrainSpec(**self.train_spec))

        # validate layer_id if provided
        if self.layer_id is not None and self.layer_id < 0:
            raise ValueError("layer_id must be >= 0.")

        # token scope cross-checks
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
        if self.token_scope == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when token_scope is 'from_position'.")
