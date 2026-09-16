"""CorDA-derived PCA arguments."""
from dataclasses import dataclass

import torch

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class CordaPCAArgs(BaseArgs):
    """Fit from paired last-token Linear inputs, keyed by exact module path.

    Args:
        positive_inputs: Paired `[N, in_features]` positive samples per module.
        negative_inputs: Negative samples with matching shapes and module paths.
        directions: Frozen output offsets, supplied instead of calibration inputs.
        rank: Leading singular modes to retain; -1 keeps all modes.
        damping: Multiplier of the calibration mean square for covariance damping.
        strength: Multiplier of the output offset; zero disables the edit.
    """

    positive_inputs: dict[str, torch.Tensor] | None = None
    negative_inputs: dict[str, torch.Tensor] | None = None
    directions: dict[str, torch.Tensor] | None = None
    rank: int = -1
    damping: float = 0.01
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.rank != -1 and self.rank < 1:
            raise ValueError("rank must be -1 or positive")
        if self.damping <= 0:
            raise ValueError("damping must be positive")
        if self.directions is None:
            if not self.positive_inputs or not self.negative_inputs:
                raise ValueError("provide paired positive_inputs and negative_inputs")
            if self.positive_inputs.keys() != self.negative_inputs.keys():
                raise ValueError("positive and negative module paths must match")
        elif not self.directions or self.positive_inputs is not None or self.negative_inputs is not None:
            raise ValueError("provide directions or calibration inputs, exclusively")
