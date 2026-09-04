import re
import string
from collections import Counter
from typing import Any

from aisteer360.evaluation.metrics.base import Metric

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = set(string.punctuation)


def _normalize_answer(text: str) -> str:
    """Apply SQuAD answer normalization (Rajpurkar et al., 2016).

    Lowercases, strips punctuation, removes the articles ``a``/``an``/``the``, and collapses
    whitespace. This is the same normalization used by the HuggingFace ``squad`` metric.

    Args:
        text (str): Raw answer text.

    Returns:
        str: The normalized answer.
    """
    text = text.lower()
    text = "".join(char for char in text if char not in _PUNCTUATION)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if the normalized prediction equals the normalized ground truth, else 0.0."""
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _token_f1(prediction: str, ground_truth: str) -> float:
    """Token-overlap F1 between the normalized prediction and ground truth.

    When either side normalizes to no tokens, returns 1.0 only if both are empty (matching the
    SQuAD reference scorer's handling of no-answer cases).
    """
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class ShortAnswerMatch(Metric):
    """SQuAD-style exact match and token-level F1 for short-answer QA.

    Implements the standard evaluation pair from SQuAD (Rajpurkar et al., 2016). Both the prediction
    and the reference are normalized (lowercased, stripped of punctuation and articles, whitespace
    collapsed); `exact_match` is then 1.0 iff the normalized strings are identical, and `f1` is the
    token-overlap F1 between them.

    F1's precision term penalizes verbose answers that merely contain the gold span (e.g.
    "The capital of France is Paris." against "Paris"), giving a smooth, non-saturating signal
    that rewards concise, correct answers.

    Each reference may be a single string or a list of acceptable strings. Scores are returned as
    fractions in `[0, 1]`.

    Rajpurkar, P., Zhang, J., Lopyrev, K. and Liang, P., 2016. SQuAD: 100,000+ questions for machine
    comprehension of text. arXiv preprint arXiv:1606.05250.
    """

    def compute(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
        references: list[str | list[str]] | None = None,
        reference_answers: list[str | list[str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute mean exact match and mean token-level F1 over a batch of responses.

        Args:
            responses (list[str]): Predicted answer strings.
            prompts (list[str] | None, optional): Unused; present for a uniform metric API.
            references (list[str | list[str]] | None, optional): Gold answer(s) per item. Each entry is
                a string or a list of acceptable strings. Either `references` or `reference_answers`
                must be supplied; they are equivalent and `references` takes precedence if both are
                given.
            reference_answers (list[str | list[str]] | None, optional): Alias for `references` (the
                toolkit's scorers pass both names).
            **kwargs: Unused.

        Returns:
            dict[str, float]: A dict with keys:

                - `"exact_match"`: mean exact match over all items, in `[0, 1]`.
                - `"f1"`: mean token-level F1 over all items, in `[0, 1]`.

        Raises:
            ValueError: If no references are supplied, if `responses` and the references differ in
                length, or if any reference is empty.
        """
        golds = references if references is not None else reference_answers
        if golds is None:
            raise ValueError("ShortAnswerMatch needs `references` (or `reference_answers`).")
        if len(responses) != len(golds):
            raise ValueError("`responses` and `references` must be the same length.")

        exact_scores: list[float] = []
        f1_scores: list[float] = []
        for response, gold in zip(responses, golds):
            response = response or ""  # treat a missing response as empty
            candidates = [gold] if isinstance(gold, str) else list(gold)
            if not candidates:
                raise ValueError("Each reference must be a non-empty string or list of strings.")
            exact_scores.append(max(_exact_match(response, candidate) for candidate in candidates))
            f1_scores.append(max(_token_f1(response, candidate) for candidate in candidates))

        count = len(exact_scores) or 1
        return {
            "exact_match": sum(exact_scores) / count,
            "f1": sum(f1_scores) / count,
        }
