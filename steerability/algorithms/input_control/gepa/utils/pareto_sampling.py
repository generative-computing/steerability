"""Frequency-weighted Pareto sampling for GEPA's parent selection (Agrawal et al., 2025, Algorithm 2).

Sample one candidate from the Pareto illumination set Ĉ, the per-instance winners pruned of dominated
candidates, with probability proportional to win-frequency `f[·]`. Falls back to uniform over the pool
only when the illumination set is empty (a degenerate score matrix).
"""
from __future__ import annotations

import random

from steerability.algorithms.input_control.common.pareto import ParetoFrontier


def sample(
    frontier: ParetoFrontier,
    pool_size: int,
    rng: random.Random | None = None,
) -> int:
    """Frequency-weighted draw over the Pareto illumination set (Algorithm 2, line 8).

    Args:
        frontier: `ParetoFrontier` over the `[n_candidates, n_instances]` score matrix.
        pool_size: Number of candidates in the pool.
        rng: Optional RNG.

    Returns:
        A candidate index in `[0, pool_size)`.
    """
    rng = rng or random
    if pool_size <= 0:
        raise ValueError("pool_size must be positive.")
    if pool_size == 1:
        return 0

    illum = frontier.illumination_set  # [(idx, coverage), ...]
    if not illum:
        return rng.randrange(pool_size)

    indices, weights = zip(*illum)
    return rng.choices(list(indices), weights=list(weights), k=1)[0]
