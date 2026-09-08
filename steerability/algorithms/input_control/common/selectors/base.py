"""Pick `k` items from a pool, optionally query-conditioned."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

T = TypeVar("T")


class BaseSelector(ABC, Generic[T]):
    """Pick `k` items from a pool, optionally query-conditioned."""

    @abstractmethod
    def select(
        self,
        items: Sequence[T],
        query: Any = None,
        k: int = 1,
        context: dict | None = None,
    ) -> list[T]:
        ...

    def prepare(
        self,
        model=None,
        tokenizer=None,
        data=None,
        **kwargs,
    ) -> None:
        """Optional offline setup (e.g. train an encoder, precompute embeddings).

        Default no-op. Parent controls invoke `selector.prepare(...)` during their own `steer()` step
        when a selector implements it.
        """
