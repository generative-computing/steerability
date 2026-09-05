"""The per-row scorer protocol consumed by steering controls."""
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class SampleScorer(Protocol):
    """Score one response against its row; higher is better.

    The row is a mapping carrying `"input"` (the query text), `"reference"` when the dataset has
    one, and any other dataset columns. Controls that optimize or rerank against a per-row score
    (prompt optimizers, sequence rerankers) accept any callable with this shape.
    """

    def __call__(self, response: str, row: Mapping[str, Any]) -> float: ...
