"""Maximum marginal relevance for diversity."""
from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np

from aisteer360.algorithms.input_control.common.selectors.base import BaseSelector

T = TypeVar("T")


@runtime_checkable
class _Encoder(Protocol):
    def encode(self, text: str) -> Any:
        ...


class MMRSelector(BaseSelector[T]):
    """Maximum-marginal-relevance selection.

    At each step, picks the item that maximizes `λ · sim(item, query) − (1 − λ) · max_j sim(item, picked_j)`.
    Used when callers want a diverse subset of candidates instead of the top-k by raw similarity.

    Args:
        encoder: Object exposing `encode(text) -> embedding` (a 1-D array-like).
        lambda_param: Trade-off in [0, 1]. 1.0 = pure relevance, 0.0 = pure diversity. Defaults to 0.5.
        item_to_text: Mapping `(item) -> str`. Default is `str()`.
    """

    def __init__(
        self,
        encoder: _Encoder,
        lambda_param: float = 0.5,
        item_to_text=None,
    ) -> None:
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError(f"lambda_param must be in [0, 1]; got {lambda_param}.")
        self.encoder = encoder
        self.lambda_param = lambda_param
        self.item_to_text = item_to_text or str

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

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

        item_embs = [np.asarray(self.encoder.encode(self.item_to_text(item))) for item in items_list]
        if query is None:
            query_emb = np.zeros_like(item_embs[0])
        else:
            query_text = query if isinstance(query, str) else self.item_to_text(query)
            query_emb = np.asarray(self.encoder.encode(query_text))

        relevance = np.array([self._cosine(query_emb, emb) for emb in item_embs])
        picked: list[int] = []
        remaining = set(range(len(items_list)))

        while remaining and len(picked) < k:
            if not picked:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                def mmr_score(i: int) -> float:
                    diversity_penalty = max(self._cosine(item_embs[i], item_embs[j]) for j in picked)
                    return self.lambda_param * relevance[i] - (1 - self.lambda_param) * diversity_penalty

                best = max(remaining, key=mmr_score)
            picked.append(best)
            remaining.discard(best)

        return [items_list[i] for i in picked]
