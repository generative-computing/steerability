"""Pick nearest neighbors of a query in a dense embedding space."""
from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np

from aisteer360.algorithms.input_control.common.selectors.base import BaseSelector

T = TypeVar("T")


@runtime_checkable
class _Encoder(Protocol):
    def encode(self, text: str) -> Any:
        ...


class DenseRetrievalSelector(BaseSelector[T]):
    """Encode a query and pick the nearest neighbors from `items` in embedding space.

    By default, items are encoded on the fly via `item_to_text`. Subclasses (e.g. EPR's selector) can
    precompute embeddings during `prepare()` and store them on the items themselves; pass
    `embedding_key` to look up a precomputed embedding from a dict-like item instead of re-encoding.

    Args:
        encoder: Object exposing `encode(text) -> embedding`.
        similarity: Either `"cosine"` (default) or `"dot"`.
        item_to_text: Mapping `(item) -> str` used when an item lacks a precomputed embedding.
        embedding_key: If items are dicts, the key under which a precomputed embedding lives.
    """

    def __init__(
        self,
        encoder: _Encoder,
        similarity: str = "cosine",
        item_to_text=None,
        embedding_key: str | None = None,
    ) -> None:
        if similarity not in ("cosine", "dot"):
            raise ValueError(f"similarity must be 'cosine' or 'dot'; got {similarity}.")
        self.encoder = encoder
        self.similarity = similarity
        self.item_to_text = item_to_text or str
        self.embedding_key = embedding_key

    def _embed_item(self, item: T) -> np.ndarray:
        if self.embedding_key is not None and isinstance(item, dict) and self.embedding_key in item:
            return np.asarray(item[self.embedding_key])
        return np.asarray(self.encoder.encode(self.item_to_text(item)))

    def _embed_query(self, query: Any) -> np.ndarray:
        if isinstance(query, np.ndarray):
            return query
        if isinstance(query, list) and query and isinstance(query[0], (float, int)):
            return np.asarray(query)
        text = query if isinstance(query, str) else self.item_to_text(query)
        return np.asarray(self.encoder.encode(text))

    def _score(self, q: np.ndarray, e: np.ndarray) -> float:
        if self.similarity == "cosine":
            denom = (np.linalg.norm(q) * np.linalg.norm(e)) or 1.0
            return float(np.dot(q, e) / denom)
        return float(np.dot(q, e))

    def _top_k_by_similarity(
        self,
        items: Sequence[T],
        query_embedding: np.ndarray,
        k: int,
    ) -> list[T]:
        items_list = list(items)
        if not items_list:
            return []
        scores = [self._score(query_embedding, self._embed_item(item)) for item in items_list]
        ranked = sorted(zip(items_list, scores), key=lambda pair: pair[1], reverse=True)
        return [item for item, _ in ranked[:k]]

    def select(
        self,
        items: Sequence[T],
        query: Any = None,
        k: int = 1,
        context: dict | None = None,
    ) -> list[T]:
        if query is None:
            raise ValueError("DenseRetrievalSelector requires a `query`.")
        query_emb = self._embed_query(query)
        return self._top_k_by_similarity(items, query_emb, k)
