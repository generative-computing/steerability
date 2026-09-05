"""LoadLoRA argument validation."""
from dataclasses import dataclass
from pathlib import Path

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class LoadLoRAArgs(BaseArgs):
    """Arguments for `LoadLoRA`.

    Attributes:
        path: Adapter directory (a PEFT `save_pretrained` output).
        base_model: Model reference the adapter was trained on; checked against the pipeline
            model at `steer()`.
        merge: Merge the adapter into the base weights after attaching.
        allow_base_mismatch: Skip the base-model check.
    """

    path: str | Path = None
    base_model: str = ""
    merge: bool = False
    allow_base_mismatch: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path is required.")
        if not self.base_model:
            raise ValueError("base_model is required.")
        self.path = str(self.path)
