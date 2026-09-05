"""CandidatePool, GEPA's genetic candidate pool with a Pareto score matrix.

Holds the candidate instructions and the `[n_candidates, n_instances]` score matrix over D_pareto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from steerability.algorithms.input_control.common.pareto import ParetoFrontier


@dataclass
class CandidatePool:
    """Genetic candidate pool with a Pareto score matrix.

    `candidates[k]` and the rows of `scores` are parallel-indexed by candidate. `scores` is
    `[n_candidates, n_instances]` over D_pareto.

    Attributes:
        candidates: Candidate instruction strings.
    """

    candidates: list[str] = field(default_factory=list)
    _scores_rows: list[list[float]] = field(default_factory=list)

    @property
    def scores(self) -> np.ndarray:
        """`[n_candidates, n_instances]` Pareto score matrix."""
        return np.array(self._scores_rows, dtype=float)

    def add(self, candidate: str, score_row: list[float]) -> int:
        """Append a candidate and return its index."""
        index = len(self.candidates)
        self.candidates.append(candidate)
        self._scores_rows.append(list(score_row))
        return index

    def frontier(self) -> "ParetoFrontier":
        from steerability.algorithms.input_control.common.pareto import ParetoFrontier
        return ParetoFrontier(self.scores)

    def best_index(self) -> int:
        """Index of the candidate with the highest mean score over D_pareto."""
        return int(self.scores.mean(axis=1).argmax())
