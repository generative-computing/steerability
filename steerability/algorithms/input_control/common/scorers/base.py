"""Score one or more prompts, optionally per-query."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class BaseScorer(ABC):
    """Score one or more prompts, optionally conditioned on a parallel sequence of queries.

    Returns a `list[float]` of length `len(prompts)`. Query-independent scorers accept `queries=None`.
    Concrete subclasses define what is scored (task evaluation, judge model, reward function). Selecting
    from those scores belongs in a `BaseSelector`.
    """

    @abstractmethod
    def score(
        self,
        prompts: Sequence[str],
        queries: Sequence[dict] | None = None,
    ) -> list[float]:
        ...
