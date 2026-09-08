"""`SequenceScorer` — the sequence-level scoring protocol.

DeAL's `reward_func` signature is adopted verbatim as the library convention, so every existing
user-supplied callable is already a `SequenceScorer`.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SequenceScorer(Protocol):
    """Score continuations of a prompt; higher = more desired.

    A callable `(prompt, continuations, params) -> list[float]`, one score per continuation.
    """

    def __call__(self, prompt: str, continuations: list[str], params: dict) -> list[float]:
        ...
