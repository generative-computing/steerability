from dataclasses import dataclass, field

from trl import SFTConfig

from steerability.algorithms.structural_control.wrappers.trl.args import TRLArgs
from steerability.algorithms.structural_control.wrappers.trl.base_mixin import resolve_config_kwargs


@dataclass
class SFTArgs(TRLArgs):
    """Arguments for SFT training via TRL's `SFTTrainer`.

    `max_length` caps the tokenized sequence length. The convenience field supplies the default; an
    entry of the same name in `training_args` overrides it.
    """

    max_length: int = field(default=4096)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.training_args.setdefault("max_length", self.max_length)
        resolve_config_kwargs(SFTConfig, self.training_args)
