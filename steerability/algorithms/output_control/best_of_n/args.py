from dataclasses import dataclass, field
from typing import Callable

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class BestOfNArgs(BaseArgs):
    """Arguments for Best-of-N sampling."""

    n: int = field(
        default=8,
        metadata={"help": "Number of full-length continuations to sample and rank."},
    )
    scorer: Callable[[str, list[str], dict], list[float]] = field(
        default=None,
        metadata={"help": "A SequenceScorer `(prompt, continuations, params) -> list[float]`; the highest-scoring sample is returned."},
    )

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"'n' must be a positive integer, got {self.n!r}.")
        if not callable(self.scorer):
            raise TypeError("'scorer' must be a callable SequenceScorer.")
