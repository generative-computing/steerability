from dataclasses import dataclass, field
from typing import Callable

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class DeALArgs(BaseArgs):

    lookahead: int = field(
        default=10,
        metadata={"help": "Number of tokens to roll out for each look-ahead beam."},
    )
    init_beams: int = field(
        default=5,
        metadata={"help": "Number of beams generated in each iteration."},
    )
    topk: int = field(
        default=3,
        metadata={"help": "The k best-scoring beams kept for the next iteration."},
    )
    max_iterations: int = field(
        default=10,
        metadata={"help": "Number of refinement steps to run."},
    )
    reward_func: Callable[[str, list[str], dict], list[float]] = field(
        default=None,
        metadata={"help": ""},  # todo: write description
    )

    # validation
    def __post_init__(self) -> None:

        for name in ("lookahead", "init_beams", "topk", "max_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.topk > self.init_beams:
            raise ValueError(
                f"topk ({self.topk}) cannot exceed init_beams ({self.init_beams})"
            )

        if not callable(self.reward_func):
            raise TypeError("`reward_func` must be a callable.")
