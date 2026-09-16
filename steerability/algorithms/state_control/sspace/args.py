"""S-space arguments."""
from dataclasses import dataclass

import torch

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class SSpaceArgs(BaseArgs):
    """Last-token Linear outputs keyed by module path, or frozen `artifacts`.

    Args:
        positive_outputs: `[N, out_features]` positive samples per module.
        negative_outputs: Negative samples; sample counts can differ.
        artifacts: Frozen bases and directions instead of calibration outputs.
        rank: Largest absolute S-space mean contrasts to retain; -1 keeps all.
        gate: Absolute cosine (`cosine`), signed cosine (`signed`), or constant (`off`).
        strength: Multiplier of the output edit; zero disables it.
    """

    positive_outputs: dict[str, torch.Tensor] | None = None
    negative_outputs: dict[str, torch.Tensor] | None = None
    artifacts: dict[str, dict[str, torch.Tensor]] | None = None
    rank: int = -1
    gate: str = "cosine"
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.rank != -1 and self.rank < 1:
            raise ValueError("rank must be -1 or positive")
        if self.gate not in ("cosine", "signed", "off"):
            raise ValueError("gate must be cosine, signed, or off")
        if self.artifacts is None:
            if not self.positive_outputs or not self.negative_outputs:
                raise ValueError("provide positive_outputs and negative_outputs")
            if self.positive_outputs.keys() != self.negative_outputs.keys():
                raise ValueError("positive and negative module paths must match")
        elif not self.artifacts or self.positive_outputs is not None or self.negative_outputs is not None:
            raise ValueError("provide artifacts or calibration outputs, exclusively")
