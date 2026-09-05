"""Route actions for `RoutedDecoding`, each lowered to a `PhasedDriver` phase plan."""
from __future__ import annotations

from dataclasses import dataclass

from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated


def _ellipsize(text: str, limit: int = 40) -> str:
    """Truncate `text` to `limit` characters with a trailing ellipsis."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class Respond:
    """Terminal canned response: splice `text` into the stream and generate nothing.

    Attributes:
        text: The canned response text, tokenized without special tokens.
    """

    text: str

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Respond requires a non-empty text.")

    def plan(self) -> list:
        """The phase plan: one appended `Fixed` phase."""
        return [Fixed(self.text, add_special_tokens=False)]

    def __str__(self) -> str:
        return f'respond("{_ellipsize(self.text)}")'


@dataclass(frozen=True)
class Prefix:
    """Canned prefix followed by generation: splice `text`, then delegate to the model.

    Attributes:
        text: The prefix text, tokenized without special tokens.
    """

    text: str

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Prefix requires a non-empty text.")

    def plan(self) -> list:
        """The phase plan: an appended `Fixed` phase, then an unbounded `Generated` phase."""
        return [Fixed(self.text, add_special_tokens=False), Generated()]

    def __str__(self) -> str:
        return f'prefix("{_ellipsize(self.text)}") + generate'


@dataclass(frozen=True)
class Generate:
    """Plain pass-through: delegate the row to the model with no splicing."""

    def plan(self) -> list:
        """The phase plan: one unbounded `Generated` phase."""
        return [Generated()]

    def __str__(self) -> str:
        return "generate"


def respond(text: str) -> Respond:
    """A terminal canned-response action carrying `text`."""
    return Respond(text)


def prefix(text: str) -> Prefix:
    """A canned-prefix-then-generate action carrying `text`."""
    return Prefix(text)


def generate() -> Generate:
    """A plain pass-through action."""
    return Generate()
