"""Score items via a `BaseScorer` and return the top-k."""
from __future__ import annotations

from typing import Any, Sequence, TypeVar

from aisteer360.algorithms.input_control.common.scorers.base import BaseScorer
from aisteer360.algorithms.input_control.common.selectors.base import BaseSelector

T = TypeVar("T")


class TopKSelector(BaseSelector[T]):
    """Pick the `k` items with the highest scores, where scores come from a `BaseScorer`.

    The scorer's `score()` is called once with the full item list. Items must be `str`-castable so they
    can be passed as `prompts` to the scorer.

    Args:
        scorer: A `BaseScorer` instance.
        item_to_prompt: Optional mapping `(item) -> str` used when items are not strings (e.g. dicts).
            Defaults to `str()`.
    """

    def __init__(self, scorer: BaseScorer, item_to_prompt=None) -> None:
        self.scorer = scorer
        self.item_to_prompt = item_to_prompt or str

    def select(
        self,
        items: Sequence[T],
        query: Any = None,
        k: int = 1,
        context: dict | None = None,
    ) -> list[T]:
        items_list = list(items)
        if not items_list:
            return []
        prompts = [self.item_to_prompt(item) for item in items_list]
        queries = None
        if query is not None:
            queries = [query if isinstance(query, dict) else {"query": query}] * len(prompts)
        scores = self.scorer.score(prompts, queries=queries)
        ranked = sorted(zip(items_list, scores), key=lambda pair: pair[1], reverse=True)
        return [item for item, _ in ranked[:k]]
