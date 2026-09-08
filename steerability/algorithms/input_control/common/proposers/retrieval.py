"""Retrieve top-n items from a corpus index given a query."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from steerability.algorithms.input_control.common.proposers.base import BaseProposer


@runtime_checkable
class _CorpusIndex(Protocol):
    """Minimal duck-typed contract for the corpus index used by `RetrievalProposer`."""

    def query(self, query_embedding, top_k: int) -> list[dict]:
        ...


@runtime_checkable
class _Encoder(Protocol):
    def encode(self, text: str) -> Any:
        ...


class RetrievalProposer(BaseProposer):
    """Encode the seed (treated as a query) and retrieve top-n items from a corpus index.

    Args:
        encoder: Object exposing `encode(text) -> embedding`.
        corpus_index: Object exposing `query(embedding, top_k) -> list[dict]`. Each returned dict should
            include at least `"text"` and `"score"`; methods may attach a `"meta"` field.
        k_max: Hard upper bound on `n` accepted by `propose`.
    """

    def __init__(self, encoder: _Encoder, corpus_index: _CorpusIndex, k_max: int = 100) -> None:
        self.encoder = encoder
        self.corpus_index = corpus_index
        self.k_max = k_max

    def propose(
        self,
        seed: Any,
        n: int = 1,
        context: dict | None = None,
    ) -> list[dict]:
        if n > self.k_max:
            raise ValueError(f"Requested n={n} exceeds k_max={self.k_max}.")
        query_text = seed if isinstance(seed, str) else str(seed)
        embedding = self.encoder.encode(query_text)
        return list(self.corpus_index.query(embedding, top_k=n))
