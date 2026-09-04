"""Selector that returns an explicit, user-specified layer."""
from .base import BaseSelector


class FixedLayerSelector(BaseSelector[int]):
    """Selects a single layer specified at construction time.

    Use when the caller already knows the target layer (e.g., from a config
    file or a previous search).

    Args:
        layer_id: The layer index to select.
    """

    def __init__(self, layer_id: int):
        if layer_id < 0:
            raise ValueError(f"layer_id must be >= 0, got {layer_id}.")
        self.layer_id = layer_id

    def select(self, *, num_layers: int) -> int:
        """Return the fixed layer id after bounds-checking.

        Args:
            num_layers: Total number of layers in the model.

        Returns:
            The stored layer id.

        Raises:
            ValueError: If layer_id >= num_layers.
        """
        if self.layer_id >= num_layers:
            raise ValueError(
                f"layer_id {self.layer_id} out of range [0, {num_layers})."
            )
        return self.layer_id
