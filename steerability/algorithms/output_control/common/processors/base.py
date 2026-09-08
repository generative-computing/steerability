"""Base class for stateful logits processors.

A processor must behave as a function of `(prefix_ids, scores)`; subclasses key any internal state
on the prefix and re-derive it on a prefix mismatch, so drivers may restart, rewind, or reorder
sequences and scoring may replay prefixes teacher-forced.
"""
from __future__ import annotations

import torch
from transformers import LogitsProcessor


class PrefixKeyedProcessor(LogitsProcessor):
    """A `LogitsProcessor` whose internal state is memoization keyed on the prefix.

    Drivers restart, rewind, and reorder sequences (segment search re-enters from a shorter
    frontier; beam reordering permutes rows; `compute_logprobs` replays prefixes teacher-forced). A
    processor holding hidden per-call state silently corrupts under any of these. This base makes
    the purity contract of `OutputControl.get_logits_processors` structural: `process()` runs only
    after the base has verified the observed `input_ids` extend the last-seen prefix row-for-row;
    otherwise `reset_state()` re-derives from scratch.

    Subclasses implement:

        - `reset_state(input_ids)`: rebuild all internal state for this (possibly new) prefix.
        - `process(input_ids, scores) -> scores`: the actual logits edit.

    Stateless subclasses implement `process` and inherit the no-op `reset_state`.
    """

    def __init__(self):
        self._last_ids: torch.Tensor | None = None

    def _extends(self, ids: torch.Tensor) -> bool:
        last = self._last_ids
        if last is None or ids.size(0) != last.size(0) or ids.size(1) < last.size(1):
            return False
        return bool(torch.equal(ids[:, : last.size(1)], last.to(ids.device)))

    def reset_state(self, input_ids: torch.Tensor) -> None:  # default: stateless
        """Rebuild internal state for a new/rewound prefix. Default no-op (stateless subclass)."""
        pass

    def process(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply the logits edit. Subclasses must override."""
        raise NotImplementedError

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if not self._extends(input_ids):
            self.reset_state(input_ids)
        out = self.process(input_ids, scores)
        self._last_ids = input_ids.detach()
        return out
