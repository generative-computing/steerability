from dataclasses import dataclass, field
from typing import Any

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class FewShotArgs(BaseArgs):
    """Arguments for few-shot input control."""

    selector: Any = field(
        default=None,
        metadata={
            "help": (
                "Selector for picking examples from the pool. May be a `BaseSelector[dict]` instance, "
                "a string name (looked up in the few-shot selector registry; e.g. 'random'), or None "
                "(defaults to `RandomSelector`)."
            )
        },
    )

    formatter: Any = field(
        default=None,
        metadata={
            "help": (
                "Formatter that renders the example block. Accepts a `BaseFormatter` instance; defaults "
                "to `FewShotBlockFormatter()` when None."
            )
        },
    )

    directive: str | None = field(
        default=None,
        metadata={"help": "Directive statement at the beginning of the system prompt."},
    )

    positive_example_pool: list[dict] | None = field(
        default=None,
        metadata={"help": "Pool of positive examples to sample from at runtime."},
    )

    negative_example_pool: list[dict] | None = field(
        default=None,
        metadata={"help": "Pool of negative examples to sample from at runtime."},
    )

    k_positive: int | None = field(
        default=None,
        metadata={"help": "Number of positive examples to sample from the pool."},
    )

    k_negative: int | None = field(
        default=None,
        metadata={"help": "Number of negative examples to sample from the pool."},
    )

    def __post_init__(self):
        if self.positive_example_pool is not None or self.negative_example_pool is not None:
            if self.k_positive is None and self.positive_example_pool:
                raise ValueError("k_positive must be specified when positive_example_pool is provided.")
            if self.k_negative is None and self.negative_example_pool:
                raise ValueError("k_negative must be specified when negative_example_pool is provided.")
