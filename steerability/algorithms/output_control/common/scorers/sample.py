"""`SampleSequenceScorer`: score continuations with a per-row `SampleScorer`."""
from __future__ import annotations

from steerability.algorithms.core.scoring import SampleScorer


class SampleSequenceScorer:
    """Score each continuation with a per-row `SampleScorer`.

    Each continuation is scored individually against a row built from the prompt and the call's
    scoring params, as `row_scorer(continuation, {"input": prompt, **params})`. The row always
    carries the prompt as `"input"`.

    Args:
        row_scorer: `SampleScorer` scoring one `(response, row)` pair; higher is better.
    """

    def __init__(self, row_scorer: SampleScorer):
        self.row_scorer = row_scorer

    def __call__(self, prompt: str, continuations: list[str], params: dict) -> list[float]:
        row = {"input": prompt, **(params or {})}
        return [float(self.row_scorer(continuation, row)) for continuation in continuations]
