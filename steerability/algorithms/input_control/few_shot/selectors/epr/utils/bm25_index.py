"""BM25 candidate retrieval for EPR labeling.

Prefers `rank_bm25` when installed; falls back to a minimal pure-NumPy implementation otherwise so
smoke tests can run on the [epr] extra-free baseline.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


class _NaiveBM25:
    """Minimal BM25 over a fixed corpus. Used when `rank_bm25` is unavailable."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = [list(toks) for toks in corpus_tokens]
        self.k1 = k1
        self.b = b
        self.N = max(len(self.corpus), 1)
        self.lengths = [len(doc) for doc in self.corpus]
        self.avgdl = (sum(self.lengths) / self.N) if self.N else 0.0
        self.doc_freq: dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        # +1 smoothing to avoid log of zero
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for doc, dl in zip(self.corpus, self.lengths):
            counts = Counter(doc)
            score = 0.0
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            for term in query_tokens:
                if term not in counts:
                    continue
                tf = counts[term]
                score += self._idf(term) * (tf * (self.k1 + 1)) / (tf + denom_norm)
            scores.append(score)
        return scores


def build_index(corpus: Sequence[str]):
    """Return an opaque index object exposing `get_scores(query_tokens) -> list[float]`."""
    tokenized = [_tokenize(doc) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
        return BM25Okapi(tokenized)
    except ImportError:
        return _NaiveBM25(tokenized)


def query(index, query_text: str, top_k: int) -> list[int]:
    """Return the indices of the top-k most BM25-relevant docs in the indexed corpus."""
    tokens = _tokenize(query_text)
    scores = list(index.get_scores(tokens))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]


def build_and_query(
    items: Sequence[dict],
    query_field: str = "output",
    candidate_set_size: int = 50,
) -> dict[int, list[int]]:
    """For each item, return BM25-nearest candidate indices in the same pool (excluding self).

    Used during EPR's offline labeling: for each training pair (x, y), retrieve a candidate set Ē of
    size L=`candidate_set_size` from the rest of the pool, ranking by BM25 over the target sequence y
    (the `query_field`).
    """
    docs = [str(item.get(query_field, "")) for item in items]
    index = build_index(docs)
    out: dict[int, list[int]] = {}
    for i, doc in enumerate(docs):
        ranked = query(index, doc, top_k=candidate_set_size + 1)
        # exclude the item itself from its own candidate set
        out[i] = [j for j in ranked if j != i][:candidate_set_size]
    return out
