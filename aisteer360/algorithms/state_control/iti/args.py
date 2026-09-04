"""ITI argument validation."""
from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.internals.data import ContrastivePairs, LabeledExamples, as_labeled_examples
from aisteer360.algorithms.state_control.common.fit_specs import VectorTrainSpec
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.token_scope import ScopeKind


@dataclass
class ITIArgs(BaseArgs):
    """Arguments for ITI (Inference-Time Intervention).

    Users provide EITHER a pre-computed steering vector OR training data.
    If data is provided, the vector is fitted during steer().

    Note:
        Direction vectors are always L2-normalized and scaled by the standard deviation
        of activations projected onto the direction (matching the paper's formula
        `alpha * sigma * theta_hat`). This calibration happens in the estimator during
        `steer()`.

    Attributes:
        steering_vector: Pre-trained steering vector with per-head directions
            (shape [num_heads, head_dim] per layer). If provided, skip training.
        data: Labeled examples (true/false statements) for training. Unlike ContrastivePairs,
            positives and negatives do not need to be equal length. Required if
            steering_vector is None.
        train_spec: Controls extraction method and accumulation mode.
        num_heads: Number of top heads to select based on probe accuracy.
            Paper default is 48 (tuned for LLaMA-7B).
        selected_heads: Override automatic head selection with explicit (layer, head) pairs.
        alpha: Scaling factor for the intervention. Paper default is 15.0 (tuned for LLaMA-7B).
        token_scope: Which tokens to steer. "after_prompt" steers only generated tokens.
        last_k: Required when token_scope == "last_k".
        from_position: Required when token_scope == "from_position".
        use_norm_preservation: If True, wrap transform in NormPreservingTransform.
    """

    # steering vector source (provide exactly one)
    steering_vector: SteeringVector | None = None
    data: LabeledExamples | dict | None = None

    # training configuration
    train_spec: VectorTrainSpec | dict = field(
        default_factory=lambda: VectorTrainSpec(method="mean_diff", accumulate="last_token")
    )

    # head selection
    num_heads: int = 48
    selected_heads: list[tuple[int, int]] | None = None

    # inference configuration
    alpha: float = 15.0
    token_scope: ScopeKind = "after_prompt"
    last_k: int | None = None
    from_position: int | None = None
    use_norm_preservation: bool = False

    def __post_init__(self):
        # exactly one of steering_vector or data must be provided
        if self.steering_vector is None and self.data is None:
            raise ValueError("Provide either steering_vector or data.")
        if self.steering_vector is not None and self.data is not None:
            raise ValueError("Provide steering_vector or data, not both.")

        # validate steering_vector if provided
        if self.steering_vector is not None:
            self.steering_vector.validate()

        # normalize dict inputs; reject ContrastivePairs with a clear error
        if self.data is not None:
            if isinstance(self.data, ContrastivePairs):
                raise TypeError(
                    "ITI requires LabeledExamples, not ContrastivePairs. "
                    "Use LabeledExamples(positives=..., negatives=...) instead. "
                    "Unlike ContrastivePairs, LabeledExamples does not require equal-length lists."
                )
            if not isinstance(self.data, LabeledExamples):
                object.__setattr__(self, "data", as_labeled_examples(self.data))

        if isinstance(self.train_spec, dict):
            object.__setattr__(self, "train_spec", VectorTrainSpec(**self.train_spec))

        # validate num_heads
        if self.num_heads < 1:
            raise ValueError("num_heads must be >= 1.")

        # validate alpha
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0.")

        # token scope cross-checks
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
        if self.token_scope == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when token_scope is 'from_position'.")
