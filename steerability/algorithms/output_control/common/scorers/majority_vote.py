"""`MajorityVoteScorer` — self-consistency scoring by answer agreement."""
from __future__ import annotations

from collections import Counter
from typing import Callable


def _final_line(text: str) -> str:
    """Default answer extractor: the last non-empty line, stripped."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


class MajorityVoteScorer:
    """Score each continuation by how many others share its extracted answer.

    Used under `BestOfN`, this is self-consistency (Wang et al. 2022): the continuation whose answer
    the plurality agrees with wins.

    Args:
        answer_extractor: `str -> str` mapping a continuation to its answer. Defaults to final-line
            extraction.
    """

    def __init__(self, answer_extractor: Callable[[str], str] | None = None):
        self.answer_extractor = answer_extractor or _final_line

    def __call__(self, prompt: str, continuations: list[str], params: dict) -> list[float]:
        answers = [self.answer_extractor(c) for c in continuations]
        counts = Counter(answers)
        # score = agreement count with the others (exclude self)
        return [float(counts[a] - 1) for a in answers]
