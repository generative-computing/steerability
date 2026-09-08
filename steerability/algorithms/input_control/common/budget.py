"""Counter for capping the number of rollouts an optimizer may consume."""
from __future__ import annotations


class RolloutBudget:
    """Counter for capping the number of rollouts an optimizer may consume.

    Used by methods that bound offline search (GEPA, CPO).

    Example:
        budget = RolloutBudget(1000)
        while budget:
            budget.charge(minibatch_size)
            ...
    """

    def __init__(self, max_rollouts: int) -> None:
        if max_rollouts < 0:
            raise ValueError(f"max_rollouts must be non-negative; got {max_rollouts}.")
        self.max = max_rollouts
        self.spent = 0

    @property
    def remaining(self) -> int:
        return max(self.max - self.spent, 0)

    def __bool__(self) -> bool:
        return self.spent < self.max

    def charge(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError(f"Cannot charge a negative amount; got {n}.")
        self.spent += n
