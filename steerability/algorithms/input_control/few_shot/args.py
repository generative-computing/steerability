from dataclasses import dataclass, field
from typing import Any

from steerability.algorithms.core.base_args import BaseArgs

_SYSTEM_MODES = frozenset({"append", "prepend", "insert"})


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
                "to `FewShotBlockFormatter(mode=system_mode, separator=separator)` when None. A supplied "
                "instance owns its own placement, so `system_mode` and `separator` are not consulted."
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

    system_mode: str = field(
        default="append",
        metadata={
            "help": (
                "How the rendered example block combines with an existing leading system message on chat input: "
                "'append' (default) joins it after the existing content, 'prepend' before it, and 'insert' adds it "
                "as a separate second system message. With no leading system message every mode inserts one system "
                "message containing the block. Some chat templates (Qwen3) reject a second system message. Not "
                "consulted when `formatter` is given."
            )
        },
    )

    separator: str = field(
        default="\n\n",
        metadata={"help": "String inserted between the existing system content and the block for 'append' and "
                          "'prepend'. Empty string allowed. Not consulted when `formatter` is given."},
    )

    def __post_init__(self):
        if self.system_mode not in _SYSTEM_MODES:
            raise ValueError(f"system_mode must be one of {sorted(_SYSTEM_MODES)}; got {self.system_mode!r}.")
        if not isinstance(self.separator, str):
            raise TypeError(f"separator must be a str; got {type(self.separator).__name__}.")
        if self.positive_example_pool is not None or self.negative_example_pool is not None:
            if self.k_positive is None and self.positive_example_pool:
                raise ValueError("k_positive must be specified when positive_example_pool is provided.")
            if self.k_negative is None and self.negative_example_pool:
                raise ValueError("k_negative must be specified when negative_example_pool is provided.")
