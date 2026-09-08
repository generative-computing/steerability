"""Selector that returns top-K heads by probe accuracy."""
import logging
from typing import ClassVar

from ..steering_vector import SteeringVector
from .base import BaseSelector

logger = logging.getLogger(__name__)


class TopKHeadSelector(BaseSelector[list[tuple[int, int]]]):
    """Selects the top-K (layer_id, head_id) pairs by probe accuracy.

    Used by ITI to select which attention heads to intervene on based on
    linear probe training accuracy scores. "Probe" here is ITI's per-head fit-time
    classifier used to rank heads, distinct from the toolkit's `Probe` detector.

    Args:
        k: Number of top heads to select.
    """

    component_kind: ClassVar[str] = "top_k_head"

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        self.k = k

    def to_config(self) -> dict:
        """The selector's serialized parameters."""
        return {"k": int(self.k)}

    @classmethod
    def from_config(cls, params: dict) -> "TopKHeadSelector":
        """Rebuild the selector from its serialized parameters."""
        return cls(k=params["k"])

    def select(self, *, steering_vector: SteeringVector) -> list[tuple[int, int]]:
        """Return the top-K (layer_id, head_id) pairs sorted by accuracy descending.

        Args:
            steering_vector: A SteeringVector with probe_accuracies populated.

        Returns:
            List of top-K (layer_id, head_id) tuples, sorted by accuracy descending.

        Raises:
            ValueError: If the steering vector has no probe_accuracies.
        """
        if steering_vector.probe_accuracies is None:
            raise ValueError("SteeringVector has no probe_accuracies for head selection.")
        ranked = sorted(steering_vector.probe_accuracies.items(), key=lambda item: item[1], reverse=True)
        selected = [key for key, _ in ranked[: self.k]]
        if len(selected) < self.k:
            logger.warning("Requested k=%d but only %d heads available.", self.k, len(selected))
        return selected
