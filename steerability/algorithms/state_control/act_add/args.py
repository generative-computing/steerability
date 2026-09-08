"""ActAdd argument validation."""
from dataclasses import dataclass

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.state_control.common.sources import ArtifactSource
from steerability.algorithms.state_control.common.steering_vector import SteeringVector


@dataclass
class ActAddArgs(BaseArgs):
    """Arguments for ActAdd (Activation Addition).

    Users provide EITHER a pre-computed steering vector OR a prompt pair.
    If prompts are provided, the vector is extracted during steer().

    Attributes:
        steering_vector: Pre-computed steering vector (positional, [T, H] per layer).
            If provided, skip extraction. The vector must be extracted at the layer-input
            boundary of the target layer, the boundary the control injects at.
        positive_prompt: Prompt representing the desired direction (e.g., "Love").
        negative_prompt: Prompt representing the opposite (e.g., "Hate").
        layer_id: Layer to inject at. If None, uses a depth-based heuristic.
        multiplier: Scaling coefficient (called ``c`` in the paper). Typical
            values range from 1 to 15 depending on model size and behavior.
        alignment: Absolute token position at which injection begins (called ``a`` in the
            paper); row ``t`` of the vector is added at position ``alignment + t``. Use 1
            to skip a BOS token when the prompt tokenization prepends one. Default: 0.
        normalize_vector: If True, L2-normalize each token position's
            direction vector independently before applying.
        use_norm_preservation: If True, wrap the transform in
            NormPreservingTransform to prevent distribution shift.
    """

    # steering vector source (provide exactly one path)
    steering_vector: "SteeringVector | ArtifactSource | None" = None
    positive_prompt: str | None = None
    negative_prompt: str | None = None

    # inference configuration
    layer_id: int | None = None
    multiplier: float = 1.0
    alignment: int = 0
    normalize_vector: bool = False
    use_norm_preservation: bool = False

    def __post_init__(self):
        # exactly one source must be provided
        has_vector = self.steering_vector is not None
        has_prompts = self.positive_prompt is not None and self.negative_prompt is not None
        if has_vector == has_prompts:
            raise ValueError("Provide either steering_vector or (positive_prompt, negative_prompt), not both.")

        if isinstance(self.steering_vector, SteeringVector):
            self.steering_vector.validate()
        elif self.steering_vector is not None and self.normalize_vector:
            raise ValueError("normalize_vector requires a concrete steering_vector or a prompt pair.")

        if self.layer_id is not None and self.layer_id < 0:
            raise ValueError("layer_id must be >= 0.")

        if self.alignment < 0:
            raise ValueError("alignment must be >= 0.")
