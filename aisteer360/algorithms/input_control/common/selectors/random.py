"""Sample k items uniformly without replacement."""
from __future__ import annotations

import random
from typing import Any, Sequence, TypeVar

from aisteer360.algorithms.input_control.common.selectors.base import BaseSelector

T = TypeVar("T")


class RandomSelector(BaseSelector[T]):
    """Sample `k` items uniformly without replacement.

    If `k >= len(items)`, returns a copy of all items in random order. The selector owns its
    RNG, so `reseed()` gives per-call determinism without disturbing other consumers of the
    process-global generator.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        """Re-seed the owned RNG for one call's deterministic selection."""
        self._rng.seed(seed)

    def select(
        self,
        items: Sequence[T],
        query: Any = None,
        k: int = 1,
        context: dict | None = None,
    ) -> list[T]:
        items_list = list(items)
        if k >= len(items_list):
            shuffled = items_list[:]
            self._rng.shuffle(shuffled)
            return shuffled
        return self._rng.sample(items_list, k)
