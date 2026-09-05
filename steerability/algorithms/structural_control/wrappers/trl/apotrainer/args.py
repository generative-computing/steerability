from dataclasses import dataclass, field

from steerability.algorithms.structural_control.wrappers.trl.dpotrainer.args import DPOArgs


@dataclass
class APOArgs(DPOArgs):
    """Arguments for APO training via TRL's `DPOTrainer`.

    APO uses one of the anchored preference losses, `"apo_zero"` or `"apo_down"`, in place of the
    sigmoid loss. `loss_type` may name that loss directly or lead a list combined with `loss_weights`
    (for example `["apo_zero", "sft"]`), in which case the first entry is the APO loss.
    """

    loss_type: str | list[str] = field(
        default="apo_zero",
        metadata={
            "help": "APO loss: 'apo_zero' or 'apo_down', optionally leading a list combined with loss_weights.",
            "choices": ["apo_zero", "apo_down"],
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        effective = self.training_args["loss_type"]
        loss_types = [effective] if isinstance(effective, str) else list(effective)
        if not loss_types or loss_types[0] not in ("apo_zero", "apo_down"):
            raise ValueError(
                f"Loss type was set to '{effective}'. It must be set to either 'apo_zero' or 'apo_down'."
            )
