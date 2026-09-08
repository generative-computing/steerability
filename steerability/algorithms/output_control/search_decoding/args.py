from dataclasses import dataclass, field
from typing import Any

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class SearchDecodingArgs(BaseArgs):
    """Arguments for the `SearchDecoding` generic (the segment-shape driver).

    The defaults are best-of-N: one full-budget segment, sampled `num_candidates` times, keeping the
    scorer's argmax. Iterative lookahead search is the same control with `segment_len`,
    `max_iterations > 1`, `keep_k > 1`, `propose_mode="beam"`.
    """

    scorer: Any = field(
        default=None,
        metadata={"help": "The sequence scorer: a SequenceScorer callable, a SampleSequenceScorer/other "
                          "instance, or a dict spec with a 'kind' key (see resolve_scorer)."},
    )
    segment_len: int | None = field(
        default=None,
        metadata={"help": "Max new tokens per rollout. None uses the call's max_new_tokens (best-of-N)."},
    )
    num_candidates: int = field(
        default=8,
        metadata={"help": "Number of continuations proposed per iteration."},
    )
    keep_k: int = field(
        default=1,
        metadata={"help": "Number of beams retained each iteration."},
    )
    max_iterations: int = field(
        default=1,
        metadata={"help": "Maximum search iterations."},
    )
    propose_mode: str = field(
        default="sample",
        metadata={"help": "Rollout mode: 'sample' or 'beam'."},
    )

    def __post_init__(self) -> None:
        if not callable(self.scorer) and not isinstance(self.scorer, dict):
            raise TypeError("'scorer' must be a SequenceScorer callable, an instance, or a dict spec.")
        if self.propose_mode not in ("sample", "beam"):
            raise ValueError(f"'propose_mode' must be 'sample' or 'beam', got {self.propose_mode!r}.")
        if not isinstance(self.num_candidates, int) or self.num_candidates < 1:
            raise ValueError(f"'num_candidates' must be a positive integer, got {self.num_candidates!r}.")
        if not isinstance(self.keep_k, int) or self.keep_k < 1:
            raise ValueError(f"'keep_k' must be a positive integer, got {self.keep_k!r}.")
        if self.keep_k > self.num_candidates:
            raise ValueError(f"'keep_k' ({self.keep_k}) cannot exceed 'num_candidates' ({self.num_candidates}).")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError(f"'max_iterations' must be a positive integer, got {self.max_iterations!r}.")
        if self.segment_len is not None and (not isinstance(self.segment_len, int) or self.segment_len < 1):
            raise ValueError(f"'segment_len' must be a positive integer when set, got {self.segment_len!r}.")
