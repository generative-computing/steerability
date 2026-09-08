"""Run the task LM on a dev set under each prompt and aggregate via a per-row scorer."""
from __future__ import annotations

import logging
from typing import Callable, Sequence

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.scoring import SampleScorer
from steerability.algorithms.input_control.common.generation import generate_with_system_prompt
from steerability.algorithms.input_control.common.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


class TaskEvaluationScorer(BaseScorer):
    """For each candidate prompt, run the task LM over a dev set and aggregate per-row scores into a
    single scalar.

    Each candidate prompt is applied as the system prompt, one response is generated per dev row,
    and the candidate's score is the mean of `row_scorer(response, row)` over the dev rows.

    Args:
        task_lm: Causal language model used to generate responses on the dev set.
        tokenizer: Tokenizer paired with `task_lm`.
        dev_set: Dataset rows; each row must contain at least the keys consumed by `format_query`
            (default: `"input"` for the user prompt) and any fields `row_scorer` reads (e.g.
            `"reference"`). Must be non-empty.
        row_scorer: `SampleScorer` scoring one `(response, row)` pair; higher is better.
        gen_kwargs: Forwarded to `task_lm.generate`.
        max_dev_size: Optional cap on the number of dev rows used per scoring call.
        format_query: Callable that turns a dev row into the user-facing query text. Defaults to
            `row -> row["input"]`.

    Raises:
        ValueError: If `dev_set` is empty.
    """

    def __init__(
        self,
        task_lm: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dev_set: Sequence[dict],
        row_scorer: SampleScorer,
        gen_kwargs: dict | None = None,
        max_dev_size: int | None = None,
        format_query: Callable[[dict], str] | None = None,
    ) -> None:
        self.task_lm = task_lm
        self.tokenizer = tokenizer
        self.dev_set = list(dev_set)
        if not self.dev_set:
            raise ValueError("dev_set must be non-empty.")
        self.row_scorer = row_scorer
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

    def score(
        self,
        prompts: Sequence[str],
        queries: Sequence[dict] | None = None,
    ) -> list[float]:
        if queries is not None:
            logger.debug("TaskEvaluationScorer ignores per-prompt `queries`; uses dev_set instead.")
        dev = self._resolve_dev()

        scores: list[float] = []
        for prompt in prompts:
            responses = self._generate_responses(prompt, dev)
            row_scores = [float(self.row_scorer(response, row)) for response, row in zip(responses, dev)]
            scores.append(sum(row_scores) / len(row_scores))
        return scores
