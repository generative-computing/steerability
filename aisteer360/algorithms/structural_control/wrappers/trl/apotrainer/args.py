from dataclasses import dataclass, field

from aisteer360.algorithms.structural_control.wrappers.trl.dpotrainer.args import DPOArgs


@dataclass
class APOArgs(DPOArgs):

    loss_type: str = field(
        default="apo_zero",
        metadata={
            "help": "Type of loss to use: 'apo_zero' or 'apo_down'.",
            "choices": ["apo_zero", "apo_down"],
        },
    )

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        if self.loss_type in ["apo_zero", "apo_down"]:
            self.training_args['loss_type'] = self.loss_type
        else:
            raise ValueError(f"Loss type was set to '{self.loss_type}'. It must be set to either 'apo_zero' or 'apo_down'.")
