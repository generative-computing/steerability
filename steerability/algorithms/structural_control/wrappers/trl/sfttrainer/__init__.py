from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.args import SFTArgs
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT

# __all__ = ["SFT", "SFTArgs"]

STEERING_METHOD = {
    "category": "structural_control",
    "name": "sft",
    "control": SFT,
    "args": SFTArgs,
}
