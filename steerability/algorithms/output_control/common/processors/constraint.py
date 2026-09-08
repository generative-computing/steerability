"""`ConstraintProcessor`, the integration point for constrained decoding.

Takes an automaton object exposing `reset(prefix_ids)` and `allowed(prefix_ids) -> LongTensor` of
permitted token ids, and masks everything else to -inf. Grammar, JSON-schema, or regex backends
plug in here by supplying the automaton (outlines-core / xgrammar adapted to the automaton
protocol); no FSM engine is built in-repo.
"""
from __future__ import annotations

from typing import Protocol

import torch

from steerability.algorithms.output_control.common.processors.base import PrefixKeyedProcessor


class ConstraintAutomaton(Protocol):
    """Minimal automaton protocol a `ConstraintProcessor` drives."""

    def reset(self, prefix_ids: torch.Tensor) -> None:
        """Reset automaton state for a new/rewound prefix."""
        ...

    def allowed(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Return a 1-D `LongTensor` of token ids permitted at the current step."""
        ...


class ConstraintProcessor(PrefixKeyedProcessor):
    """Mask all logits except the automaton's currently permitted token ids.

    Args:
        automaton: An object implementing the `ConstraintAutomaton` protocol.
    """

    def __init__(self, automaton: ConstraintAutomaton):
        super().__init__()
        self.automaton = automaton

    def reset_state(self, input_ids: torch.Tensor) -> None:
        self.automaton.reset(input_ids)

    def process(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        allowed = self.automaton.allowed(input_ids).to(scores.device)
        out = torch.full_like(scores, float("-inf"))
        out[:, allowed] = scores[:, allowed]
        return out
