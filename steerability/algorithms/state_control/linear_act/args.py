"""Linear-AcT arguments."""
from dataclasses import dataclass

import torch

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class LinearAcTArgs(BaseArgs):
    """Last-token decoder outputs `[N, hidden_size]`, keyed by layer index.

    Args:
        positive_activations: Target distribution samples per layer.
        negative_activations: Source samples with matching shapes and layers.
        affine: Frozen `[2, hidden_size]` slope/bias tensors instead of samples.
        strength: Interpolation from identity (0) to transport (1); extrapolation allowed.
    """

    positive_activations: dict[int, torch.Tensor] | None = None
    negative_activations: dict[int, torch.Tensor] | None = None
    affine: dict[int, torch.Tensor] | None = None
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.affine is None:
            if not self.positive_activations or not self.negative_activations:
                raise ValueError("provide positive_activations and negative_activations")
            if self.positive_activations.keys() != self.negative_activations.keys():
                raise ValueError("positive and negative layers must match")
            keys = self.positive_activations
        else:
            if not self.affine or self.positive_activations is not None or self.negative_activations is not None:
                raise ValueError("provide affine or calibration activations, exclusively")
            keys = self.affine
        if any(not isinstance(k, int) or k < 0 for k in keys):
            raise ValueError("layer indices must be nonnegative integers")
