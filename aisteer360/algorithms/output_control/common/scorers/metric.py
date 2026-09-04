"""`MetricScorer` — drive decoding toward the metric that benchmarks it.

Adapts any `evaluation.metrics.base.Metric` into a `SequenceScorer`: the same metric used to
*benchmark* a behavior can *drive* decoding toward it. Each continuation is scored individually and
a named scalar is read from the metric's result dict.
"""
from __future__ import annotations

from aisteer360.evaluation.metrics.base import Metric


class MetricScorer:
    """Score each continuation with an `evaluation` `Metric`.

    Args:
        metric: The metric to apply.
        score_key: Key to read from the metric's result dict as the scalar score.
        pass_prompt: When True, pass the prompt as the metric's `prompts` argument.
    """

    def __init__(self, metric: Metric, score_key: str, pass_prompt: bool = False):
        self.metric = metric
        self.score_key = score_key
        self.pass_prompt = pass_prompt

    def __call__(self, prompt: str, continuations: list[str], params: dict) -> list[float]:
        scores: list[float] = []
        for continuation in continuations:
            kwargs = {"prompts": [prompt]} if self.pass_prompt else {}
            result = self.metric.compute([continuation], **kwargs)
            value = result[self.score_key]
            if isinstance(value, (list, tuple)):
                value = value[0]
            scores.append(float(value))
        return scores
