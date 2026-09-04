"""
Base classes for evaluation metrics.

Contains two classes:

- `Metric`: Base class for all evaluation metrics.
- `LLMJudgeMetric`: Base class for LLM-as-a-judge metrics (subclasses `Metric`)
"""
from aisteer360.evaluation.metrics.backend_utils import release_metric_backends
from aisteer360.evaluation.metrics.base_judge import LLMJudgeMetric
from aisteer360.evaluation.metrics.generic.factuality import Factuality
from aisteer360.evaluation.metrics.generic.perplexity import Perplexity
from aisteer360.evaluation.metrics.generic.relevance import Relevance
from aisteer360.evaluation.metrics.generic.short_answer_match import ShortAnswerMatch

from .base import Metric

__all__ = [
    "Metric",
    "LLMJudgeMetric",
    "Factuality",
    "Perplexity",
    "Relevance",
    "ShortAnswerMatch",
    "release_metric_backends"
]
