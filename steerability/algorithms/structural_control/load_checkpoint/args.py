"""LoadCheckpoint argument validation."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class LoadCheckpointArgs(BaseArgs):
    """Arguments for `LoadCheckpoint`.

    Attributes:
        path: Checkpoint directory (a local `save_pretrained` output or a Hub id).
        device_map: Device map forwarded to `from_pretrained`.
        hf_model_kwargs: Extra keyword arguments forwarded to `from_pretrained`.
        trust_remote_code: Trust remote code when loading the checkpoint.
    """

    path: str | Path = None
    device_map: str | dict = "auto"
    hf_model_kwargs: dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path is required.")
        self.path = str(self.path)
