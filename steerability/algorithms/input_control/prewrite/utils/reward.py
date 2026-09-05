"""Reward-function adapters for GRPO-trained PRewrite rewriters.

Wraps a `TaskEvaluationScorer` into the callable shape TRL's `GRPOTrainer` expects,
`reward_func(prompts, completions, **kwargs)`, returning one float per completion. The reward applies
the rewritten instruction with the frozen task model over a dev set and scores the answers with a
per-row `SampleScorer` (Kong et al., 2024).

For a callable reward on trl==0.16.1, `GRPOTrainer` calls
`reward_func(prompts=prompts, completions=completions, **reward_kwargs)`. Completions are plain strings
when the dataset `"prompt"` column is text (the PRewrite case) and conversational message lists when it
is a chat. Both shapes are handled here.
"""
from __future__ import annotations

from typing import Any, Callable

from steerability.algorithms.input_control.common.scorers.task_evaluation import TaskEvaluationScorer


def _completion_text(completion: Any) -> str:
    """Extract raw text from a GRPO completion (plain string or conversational message list)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    return str(completion)


def make_scorer_reward_func(
    scorer: TaskEvaluationScorer,
    parse_fn: Callable[[str], list[str]] | None = None,
) -> Callable:
    """Wrap a `TaskEvaluationScorer` as a GRPO reward function.

    Each completion is a candidate rewritten instruction; its reward is the scorer's dev-set score for
    that instruction. Identical rewrites are scored once and the result reused. PRewrite rewrites are
    input-agnostic (the rewriter sees only the seed meta-prompt, not the task inputs), so a generation
    group often contains duplicates.

    Args:
        scorer: Configured `TaskEvaluationScorer`. `scorer.score(list_of_rewrites)` returns one float per
            rewrite (a full dev-set pass per rewrite under the frozen task model).
        parse_fn: Optional parser applied to each completion before scoring (e.g.
            `parse_concise_instruction`), keeping the reward consistent with how the proposer parses
            candidates at inference time. If it returns an empty list for a completion, the raw stripped
            completion text is scored instead.

    Returns:
        A callable `reward_func(prompts, completions, **kwargs) -> list[float]` aligned to `completions`.
    """

    def _extract(completion: Any) -> str:
        text = _completion_text(completion).strip()
        if parse_fn is not None:
            parsed = parse_fn(text)
            if parsed:
                return parsed[0].strip()
        return text

    def reward_func(prompts, completions, **kwargs) -> list[float]:
        rewrites = [_extract(completion) for completion in completions]
        unique = list(dict.fromkeys(rewrites))  # order-preserving dedup
        if not unique:
            return []
        scores = scorer.score(unique)
        lookup = dict(zip(unique, scores))
        return [lookup[rewrite] for rewrite in rewrites]

    return reward_func
