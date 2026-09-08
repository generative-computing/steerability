"""Constrained decoding argument validation."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.execution.payloads import ConstraintSource, as_constraint_source


@dataclass
class ConstrainedDecodingArgs(BaseArgs):
    """Arguments for constrained decoding.

    The constraint is given either declaratively (a `ConstraintSource`, or one of the
    convenience fields `json_schema`, `regex`, `grammar`, `choice`) or as a live automaton
    object. A declarative constraint renders per execution arm; an automaton object has no
    declarative form and runs in process only.

    Attributes:
        source: The declarative constraint, or a mapping with `kind` and `value` keys.
        json_schema: Convenience for `ConstraintSource(kind="json_schema", value=...)`.
        regex: Convenience for `ConstraintSource(kind="regex", value=...)`.
        grammar: Convenience for `ConstraintSource(kind="grammar", value=...)` (EBNF).
        choice: Convenience for `ConstraintSource(kind="choice", value=...)`.
        automaton: A live object implementing the `ConstraintAutomaton` protocol
            (`reset(prefix_ids)` and `allowed(prefix_ids)`).
        include_in_scoring: Whether the constraint participates in `compute_logprobs`.
            Structured outputs do not apply to prompt logprobs, so scoring with the constraint
            enabled requires the in-process backend; False opts out of scoring entirely.
    """

    source: ConstraintSource | Mapping | None = None
    json_schema: str | Mapping | None = None
    regex: str | None = None
    grammar: str | None = None
    choice: Sequence[str] | None = None
    automaton: Any | None = None
    include_in_scoring: bool = True

    def __post_init__(self):
        convenience = {
            "json_schema": self.json_schema,
            "regex": self.regex,
            "grammar": self.grammar,
            "choice": self.choice,
        }
        supplied = [name for name, value in convenience.items() if value is not None]
        given = int(self.source is not None) + int(self.automaton is not None) + len(supplied)
        if given != 1:
            raise ValueError(
                "Provide exactly one constraint: source, automaton, or one of "
                "json_schema/regex/grammar/choice."
            )
        if self.source is not None:
            object.__setattr__(self, "source", as_constraint_source(self.source))
        elif supplied:
            kind = supplied[0]
            object.__setattr__(
                self, "source", ConstraintSource(kind=kind, value=convenience[kind])
            )
