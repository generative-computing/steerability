"""Predicates and ordered routes over named per-row boolean decisions.

A `Predicate` is a boolean expression over decision names, built from `P(name)` leaves and the
operators `&` (and), `|` (or), and `~` (not), and evaluated per row against a mapping from
decision name to per-row values. Probes are the canonical producer of such decisions, but any
named boolean source works. `Router` holds an ordered list of `Route`s with first-match-wins
semantics, evaluated independently per row, so one batched call can route each prompt to a
different route.

Actions are opaque payloads: a `Route` carries whatever the consumer interprets (e.g. a decoding
driver lowers actions to phase plans). This module depends only on `torch` and the standard
library.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


def _normalize_decisions(decisions: Mapping[str, torch.Tensor | bool]) -> tuple[dict[str, torch.Tensor], int]:
    """Validate a decisions mapping and normalize every value to a 1-D bool tensor.

    Each value must be a 1-D bool tensor of a common length `num_rows`, or a bare python bool,
    which is accepted only when `num_rows == 1` (the single-row scalar allowance).

    Args:
        decisions: Mapping from decision name to per-row decisions.

    Returns:
        Tuple of (normalized mapping, num_rows).

    Raises:
        ValueError: If `decisions` is empty, a tensor is not 1-D or not bool dtype, tensor
            lengths disagree, or a bare bool is mixed with multi-row tensors.
    """
    if not decisions:
        raise ValueError("decisions mapping is empty; at least one named decision is required.")

    num_rows: int | None = None
    for name, value in decisions.items():
        if isinstance(value, bool):
            continue
        t = torch.as_tensor(value)
        if t.dtype != torch.bool:
            raise ValueError(
                f"Decision '{name}' has dtype {t.dtype}; decisions must be bool tensors."
            )
        if t.ndim != 1:
            raise ValueError(
                f"Decision '{name}' has shape {tuple(t.shape)}; decisions must be 1-D "
                f"per-row tensors of shape [num_rows]."
            )
        if num_rows is None:
            num_rows = t.numel()
        elif t.numel() != num_rows:
            raise ValueError(
                f"Decision '{name}' has {t.numel()} rows but earlier decisions have "
                f"{num_rows}; all decisions must describe the same logical batch."
            )

    if num_rows is None:  # every value was a bare bool
        num_rows = 1

    normalized: dict[str, torch.Tensor] = {}
    for name, value in decisions.items():
        if isinstance(value, bool):
            if num_rows != 1:
                raise ValueError(
                    f"Decision '{name}' is a bare bool but the batch has {num_rows} "
                    f"rows; bare bools are accepted only when num_rows == 1."
                )
            normalized[name] = torch.tensor([value], dtype=torch.bool)
        else:
            normalized[name] = torch.as_tensor(value)
    return normalized, num_rows


class Predicate(ABC):
    """Boolean expression over named per-row decisions, evaluated per row.

    Leaves are created with `P(name)`; composites are built with the operators `&` (and),
    `|` (or), and `~` (not). `evaluate()` takes a mapping from decision name to per-row values
    (1-D bool tensors of a common length, or a bare bool for single-row batches) and returns a
    bool tensor of shape `[num_rows]`. `repr()` renders the expression infix, e.g.
    ``(legal & ~advice)``.
    """

    def evaluate(self, decisions: Mapping[str, torch.Tensor | bool]) -> torch.Tensor:
        """Evaluate the predicate against per-row decisions.

        Args:
            decisions: Mapping from decision name to a 1-D bool tensor of shape `[num_rows]`
                (a bare bool is accepted only when `num_rows == 1`).

        Returns:
            Bool tensor of shape `[num_rows]`; True where the predicate holds for that row.

        Raises:
            KeyError: If the predicate references a decision name absent from `decisions`; the
                message lists the available names.
            ValueError: If the decision tensors are malformed (see `evaluate` requirements).
        """
        normalized, _ = _normalize_decisions(decisions)
        return self._eval(normalized)

    @abstractmethod
    def _eval(self, decisions: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate against an already-normalized decisions mapping."""

    @abstractmethod
    def decision_names(self) -> set[str]:
        """The set of decision names this predicate references."""

    def __and__(self, other: "Predicate") -> "Predicate":
        if not isinstance(other, Predicate):
            return NotImplemented
        return _And(self, other)

    def __or__(self, other: "Predicate") -> "Predicate":
        if not isinstance(other, Predicate):
            return NotImplemented
        return _Or(self, other)

    def __invert__(self) -> "Predicate":
        return _Not(self)


class _Decision(Predicate):
    """Leaf predicate: the named decision's per-row value."""

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError(f"Decision name must be a non-empty string; got {name!r}.")
        self.name = name

    def _eval(self, decisions: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.name not in decisions:
            raise KeyError(
                f"Unknown decision name '{self.name}'; available decisions: {sorted(decisions)}."
            )
        return decisions[self.name]

    def decision_names(self) -> set[str]:
        return {self.name}

    def __repr__(self) -> str:
        return self.name


class _And(Predicate):
    def __init__(self, left: Predicate, right: Predicate):
        self.left = left
        self.right = right

    def _eval(self, decisions: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.left._eval(decisions) & self.right._eval(decisions)

    def decision_names(self) -> set[str]:
        return self.left.decision_names() | self.right.decision_names()

    def __repr__(self) -> str:
        return f"({self.left!r} & {self.right!r})"


class _Or(Predicate):
    def __init__(self, left: Predicate, right: Predicate):
        self.left = left
        self.right = right

    def _eval(self, decisions: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.left._eval(decisions) | self.right._eval(decisions)

    def decision_names(self) -> set[str]:
        return self.left.decision_names() | self.right.decision_names()

    def __repr__(self) -> str:
        return f"({self.left!r} | {self.right!r})"


class _Not(Predicate):
    def __init__(self, operand: Predicate):
        self.operand = operand

    def _eval(self, decisions: dict[str, torch.Tensor]) -> torch.Tensor:
        return ~self.operand._eval(decisions)

    def decision_names(self) -> set[str]:
        return self.operand.decision_names()

    def __repr__(self) -> str:
        return f"~{self.operand!r}"


def P(name: str) -> Predicate:
    """Leaf predicate over the named decision.

    Args:
        name: The decision name, matching a key of the decisions mapping at evaluation time.

    Returns:
        A `Predicate` that reads the named per-row decision.
    """
    return _Decision(name)


@dataclass(frozen=True)
class Route:
    """One named route: a predicate and the action to take when it matches.

    Attributes:
        name: Route name, unique within a `Router`; keys diagnostics and per-call action
            overrides.
        when: The predicate that must hold for a row to match this route.
        action: Opaque payload interpreted by the consumer (e.g. lowered to a phase plan by a
            decoding driver).
    """

    name: str
    when: Predicate
    action: Any

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Route name must be a non-empty string; got {self.name!r}.")
        if not isinstance(self.when, Predicate):
            raise TypeError(
                f"Route '{self.name}': `when` must be a Predicate (build one with P(name) "
                f"and &, |, ~); got {type(self.when).__name__}."
            )


def _action_label(action: Any) -> str:
    """Compact display label for an action: `str(action)` when the type defines its own
    `__str__`, else the type name (`"None"` for a missing action)."""
    if action is None:
        return "None"
    if type(action).__str__ is not object.__str__:
        return str(action)
    return type(action).__name__


class Router:
    """Ordered routes with first-match-wins semantics, evaluated independently per row.

    Each route pairs a `Predicate` with an action payload. `route()` evaluates every route's
    predicate over all rows, then assigns each row the first route whose predicate holds; rows
    matching no route fall to the default (returned as None, with `default_action` available to
    the consumer).

    Args:
        routes: Ordered routes; earlier routes take precedence. Names must be unique.
        default_action: Action payload for rows matching no route. Opaque, like `Route.action`.

    Raises:
        ValueError: If two routes share a name.
        TypeError: If an entry is not a `Route`.
    """

    def __init__(self, routes: Sequence[Route], default_action: Any = None):
        routes = tuple(routes)
        for entry in routes:
            if not isinstance(entry, Route):
                raise TypeError(f"Router entries must be Route instances; got {type(entry).__name__}.")
        seen: set[str] = set()
        for entry in routes:
            if entry.name in seen:
                raise ValueError(f"Duplicate route name '{entry.name}'; route names must be unique.")
            seen.add(entry.name)
        self.routes = routes
        self.default_action = default_action

    def route(self, decisions: Mapping[str, torch.Tensor | bool]) -> list[Route | None]:
        """Match each row to its first satisfied route.

        Every route's predicate is evaluated once over all rows, then each row takes the first
        route whose predicate holds for it. There is no batch-wide short-circuit; rows are
        routed independently.

        Args:
            decisions: Mapping from decision name to per-row decisions (see
                `Predicate.evaluate`).

        Returns:
            One entry per row: the matched `Route`, or None for rows matching no route (the
            default route).

        Raises:
            KeyError: If a route references a decision name absent from `decisions`.
            ValueError: If the decision tensors are malformed.
        """
        normalized, num_rows = _normalize_decisions(decisions)
        masks = [entry.when._eval(normalized) for entry in self.routes]
        matched: list[Route | None] = []
        for row in range(num_rows):
            match: Route | None = None
            for entry, mask in zip(self.routes, masks):
                if bool(mask[row]):
                    match = entry
                    break
            matched.append(match)
        return matched

    def decision_names(self) -> set[str]:
        """The union of decision names referenced by all routes."""
        names: set[str] = set()
        for entry in self.routes:
            names |= entry.when.decision_names()
        return names

    def validate_names(self, available: set[str]) -> None:
        """Check that every referenced decision name exists among `available`.

        Args:
            available: The decision names the consumer provides at routing time.

        Raises:
            ValueError: If any route references a decision name absent from `available`; the
                message names the missing decisions.
        """
        missing = sorted(self.decision_names() - set(available))
        if missing:
            raise ValueError(
                f"Routes reference unknown decision name(s) {missing}; available decisions: "
                f"{sorted(available)}."
            )

    def describe(self) -> str:
        """Render the route set as a plain-text flowchart.

        One line per route, in precedence order, followed by the default line. The action
        column uses `str(action)` when the action type defines its own `__str__`, else the
        type name.

        Returns:
            The multi-line flowchart string.
        """
        name_width = max((len(entry.name) for entry in self.routes), default=0)
        name_width = max(name_width, len("default"))
        pred_reprs = [repr(entry.when) for entry in self.routes]
        pred_width = max((len(p) for p in pred_reprs), default=0)
        index_width = len(str(len(self.routes))) if self.routes else 1
        body_width = index_width + 2 + name_width + 3 + 3 + pred_width + 3

        lines = ["Router"]
        for i, (entry, pred) in enumerate(zip(self.routes, pred_reprs), start=1):
            body = f"{i:>{index_width}}. {entry.name:<{name_width}}   if {pred:<{pred_width}}   "
            lines.append(f"├─ {body}-> {_action_label(entry.action)}")
        lines.append(f"└─ {'default':<{body_width}}-> {_action_label(self.default_action)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        route_names = ", ".join(entry.name for entry in self.routes)
        return f"Router([{route_names}], default_action={_action_label(self.default_action)})"
