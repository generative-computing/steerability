"""Produce n candidate items from a seed."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseProposer(ABC):
    """Produce `n` candidate items from a seed."""

    @abstractmethod
    def propose(
        self,
        seed: Any,
        n: int = 1,
        context: dict | None = None,
    ) -> list[Any]:
        ...
