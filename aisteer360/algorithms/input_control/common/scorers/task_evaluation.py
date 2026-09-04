"""Run the task LM on a dev set under each prompt and aggregate via a Metric."""
from __future__ import annotations

import logging
from typing import Any, Sequence

from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.input_control.common.generation import generate_with_system_prompt
from aisteer360.algorithms.input_control.common.scorers.base import BaseScorer
from aisteer360.evaluation.metrics.base import Metric

logger = logging.getLogger(__name__)


class TaskEvaluationScorer(BaseScorer):
    """For each candidate prompt, run the task LM over a dev set and aggregate per-instance results into a
    single scalar via a `Metric`.

    Args:
        task_lm: Causal language model used to generate responses on the dev set.
        tokenizer: Tokenizer paired with `task_lm`.
        dev_set: Dataset rows; each row must contain at least the keys consumed by `format_query` (default:
            `"input"` for the user prompt and optionally `"reference"` / other fields the metric reads).
        metric: An `aisteer360.evaluation.metrics.base.Metric`; the aggregate scalar returned by
            `metric.compute(...)` (or its first numeric value, if `compute` returns a dict) is used.
        score_key: When the metric returns a dict, which key to extract. If None, picks the first
            numeric value in iteration order.
        gen_kwargs: Forwarded to `task_lm.generate`.
        max_dev_size: Optional cap on the number of dev rows used per scoring call.
        format_query: Callable that turns a dev row into the user-facing query text. Defaults to
            `row -> row["input"]`.
    """

    def __init__(
        self,
        task_lm,
        tokenizer: PreTrainedTokenizerBase,
        dev_set: Sequence[dict],
        metric: Metric,
        score_key: str | None = None,
        gen_kwargs: dict | None = None,
        max_dev_size: int | None = None,
        format_query=None,
    ) -> None:
        self.task_lm = task_lm
        self.tokenizer = tokenizer
        self.dev_set = list(dev_set)
        self.metric = metric
        self.score_key = score_key
        self.gen_kwargs = gen_kwargs or {"max_new_tokens": 32, "do_sample": False}
        self.max_dev_size = max_dev_size
        self.format_query = format_query or (lambda row: row["input"])

    def _resolve_dev(self) -> list[dict]:
        if self.max_dev_size is not None:
            return self.dev_set[: self.max_dev_size]
        return self.dev_set

    def _generate_responses(self, prompt: str, dev_rows: list[dict]) -> list[str]:
        """Generate responses for all dev rows under a single candidate prompt."""
        queries = [self.format_query(row) for row in dev_rows]
        return generate_with_system_prompt(
            self.task_lm, self.tokenizer, prompt, queries, gen_kwargs=self.gen_kwargs
        )

    def _aggregate(self, metric_result: Any) -> float:
        if isinstance(metric_result, dict):
            if self.score_key is not None:
                value = metric_result[self.score_key]
            else:
                value = next(
                    (v for v in metric_result.values() if isinstance(v, (int, float))),
                    None,
                )
                if value is None:
                    raise ValueError(
                        f"Metric returned dict {metric_result!r} with no numeric value; "
                        "set `score_key` to disambiguate."
                    )
            return float(value)
        if isinstance(metric_result, (int, float)):
            return float(metric_result)
        raise TypeError(f"Cannot interpret metric result {metric_result!r} as scalar.")

    def score(
        self,
        prompts: Sequence[str],
        queries: Sequence[dict] | None = None,
    ) -> list[float]:
        if queries is not None:
            logger.debug("TaskEvaluationScorer ignores per-prompt `queries`; uses dev_set instead.")
        dev = self._resolve_dev()

        query_texts = [self.format_query(row) for row in dev]
        references = [row.get("reference") for row in dev]
        has_references = any(r is not None for r in references)

        scores: list[float] = []
        for prompt in prompts:
            responses = self._generate_responses(prompt, dev)

            metric_kwargs: dict[str, Any] = {"prompts": query_texts}
            if has_references:
                metric_kwargs["reference_answers"] = references
                metric_kwargs["references"] = references
            result = self.metric.compute(responses, **metric_kwargs)
            scores.append(self._aggregate(result))
        return scores
