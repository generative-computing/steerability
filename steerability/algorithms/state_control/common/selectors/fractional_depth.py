"""Selector that picks a layer at a given fractional depth of the model."""
from typing import ClassVar

from .base import BaseSelector


class FractionalDepthSelector(BaseSelector[int]):
    """Selects a layer at a specified fraction of total model depth.

    For example, ``FractionalDepthSelector(0.4)`` on a 32-layer model returns
    layer 12 (≈ 40% depth), matching the CAA paper's finding that layer 13/32
    is near-optimal.

    Args:
        fraction: Target depth as a fraction in (0, 1).
        minimum: Floor for the returned layer id (default 0). Useful when
            very early layers should be excluded (e.g., ActAdd avoids layer 0).
    """

    component_kind: ClassVar[str] = "fractional_depth"

    def __init__(self, fraction: float, minimum: int = 0):
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {fraction}.")
        if minimum < 0:
            raise ValueError(f"minimum must be >= 0, got {minimum}.")
        self.fraction = fraction
        self.minimum = minimum

    def to_config(self) -> dict:
        """The selector's serialized parameters."""
        return {"fraction": float(self.fraction), "minimum": int(self.minimum)}

    @classmethod
    def from_config(cls, params: dict) -> "FractionalDepthSelector":
        """Rebuild the selector from its serialized parameters."""
        return cls(fraction=params["fraction"], minimum=params.get("minimum", 0))

    def select(self, *, num_layers: int) -> int:
        """Compute and return the target layer id.

        Args:
            num_layers: Total number of layers in the model.

        Returns:
            Layer index at approximately ``fraction * num_layers``, clamped
            to ``[minimum, num_layers - 1]``.
        """
        layer_id = int(num_layers * self.fraction)
        layer_id = max(self.minimum, min(layer_id, num_layers - 1))
        return layer_id


class LateThirdSelector(BaseSelector[list[int]]):
    """Selects the late third of the model's layers.

    The default behavior-layer heuristic for conditional activation steering.
    """

    component_kind: ClassVar[str] = "late_third"

    def to_config(self) -> dict:
        """The selector's serialized parameters (none)."""
        return {}

    @classmethod
    def from_config(cls, params: dict) -> "LateThirdSelector":
        """Rebuild the selector from its serialized parameters."""
        return cls()

    def select(self, *, num_layers: int) -> list[int]:
        """Return the last third of the layer indices.

        Args:
            num_layers: Total number of layers in the model.

        Returns:
            The selected layer ids, ascending.
        """
        from .utils.layer_heuristics import late_third

        return late_third(num_layers)
