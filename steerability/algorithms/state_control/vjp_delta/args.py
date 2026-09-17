"""Arguments for VJP-delta steering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.internals.data import LabeledExamples, as_labeled_examples
from steerability.algorithms.state_control.common.sources import ArtifactSource
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.token_scope import ScopeKind


@dataclass
class VJPDeltaArgs(BaseArgs):
    """Arguments for `VJPDelta`.

    Args:
        steering_vector: A precomputed `SteeringVector` skips VJP fitting. An `ArtifactSource` resolves its own artifact.
        data: Independent positive and negative raw prompt pools for `VJPDeltaFit`.
        target_layer: Target layer for the contrast. None selects `num_layers - 3`.
        source_layer_ids: Source layers for VJPs. None selects every layer before the target.
        skip_first: Prefix positions excluded during gradient extraction.
        max_length: Maximum tokenized fit-prompt length.
        batch_size: Fit prompts per differentiable forward.
        strength: Multiplier used when applying the normalized directions.
        token_scope: Positions that receive the additive intervention during generation.
        last_k: Required with `token_scope="last_k"`.
        from_position: Required with `token_scope="from_position"`.
    """

    steering_vector: SteeringVector | ArtifactSource | None = None
    data: LabeledExamples | dict | None = None
    target_layer: int | None = None
    source_layer_ids: Sequence[int] | None = None
    skip_first: int = 16
    max_length: int = 384
    batch_size: int = 8
    strength: float = 1.0
    token_scope: ScopeKind = "after_prompt"
    last_k: int | None = None
    from_position: int | None = None

    def __post_init__(self) -> None:
        if (self.steering_vector is None) == (self.data is None):
            raise ValueError("Provide exactly one of steering_vector or data.")
        if isinstance(self.steering_vector, SteeringVector):
            self.steering_vector.validate()
        if self.data is not None and not isinstance(self.data, LabeledExamples):
            self.data = as_labeled_examples(self.data)
        if self.target_layer is not None and self.target_layer < 0:
            raise ValueError("target_layer must be >= 0.")
        if self.source_layer_ids is not None:
            source_ids = tuple(int(layer_id) for layer_id in self.source_layer_ids)
            if not source_ids or min(source_ids) < 0 or len(set(source_ids)) != len(source_ids):
                raise ValueError("source_layer_ids must be a non-empty sequence of unique integers >= 0.")
            self.source_layer_ids = source_ids
        if self.skip_first < 0:
            raise ValueError("skip_first must be >= 0.")
        if self.max_length < 2:
            raise ValueError("max_length must be >= 2.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
        if self.token_scope == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when token_scope is 'from_position'.")
