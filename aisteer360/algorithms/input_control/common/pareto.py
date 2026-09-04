"""Utility for Pareto analysis over a [n_candidates, n_instances] score matrix.

Used by methods that maintain instance-wise best-so-far records during search (GEPA).
"""
from __future__ import annotations

import numpy as np


class ParetoFrontier:
    """Pareto-frontier utility over a `[n_candidates, n_instances]` score matrix.

    Maximizes scores by default; pass `minimize=True` for the opposite. Scores are handled internally as a
    maximization problem.
    """

    def __init__(self, scores: np.ndarray, minimize: bool = False) -> None:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2:
            raise ValueError(f"scores must be 2-D [n_candidates, n_instances]; got shape {scores.shape}.")
        if scores.size == 0:
            raise ValueError("scores must be non-empty.")
        self._scores = -scores if minimize else scores
        self._minimize = minimize

    @property
    def per_instance_best(self) -> np.ndarray:
        """Per-instance maximum score across candidates. Shape `[n_instances]`. Returned in the original
        sign convention (i.e. minimum, when `minimize=True`)."""
        best = self._scores.max(axis=0)
        return -best if self._minimize else best

    @property
    def per_instance_winners(self) -> list[set[int]]:
        """For each instance, the set of candidate indices that achieve the per-instance best.

        Multiple candidates may tie.
        """
        best = self._scores.max(axis=0, keepdims=True)
        wins = self._scores == best  # [n_candidates, n_instances]
        return [set(np.where(wins[:, j])[0].tolist()) for j in range(wins.shape[1])]

    @property
    def non_dominated(self) -> list[int]:
        """Candidate indices that are not strictly dominated by any other candidate.

        Candidate `a` is dominated by `b` iff `b >= a` on every instance and `b > a` on at least one.
        """
        n = self._scores.shape[0]
        keep: list[int] = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(self._scores[j] >= self._scores[i]) and np.any(self._scores[j] > self._scores[i]):
                    dominated = True
                    break
            if not dominated:
                keep.append(i)
        return keep

    def coverage(self, candidate: int) -> int:
        """Number of instances on which `candidate` is a per-instance winner (ties count)."""
        winners = self.per_instance_winners
        return sum(1 for s in winners if candidate in s)

    @property
    def illumination_set(self) -> list[tuple[int, int]]:
        """Algorithm 2's pruned candidate set Ĉ with win-frequencies f[·].

        Returns `(candidate_index, coverage)` for every candidate that (a) is a per-instance winner on
        at least one instance and (b) is not strictly dominated by any other candidate. Equivalent to
        `[(k, self.coverage(k)) for k in self.non_dominated if self.coverage(k) >= 1]`.

        Empty only in degenerate cases; callers should fall back to uniform sampling then.
        """
        result: list[tuple[int, int]] = []
        for k in self.non_dominated:
            cov = self.coverage(k)
            if cov >= 1:
                result.append((k, cov))
        return result
