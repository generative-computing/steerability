from aisteer360.algorithms.structural_control.wrappers.trl.grpotrainer.args import GRPOArgs
from aisteer360.algorithms.structural_control.wrappers.trl.grpotrainer.control import GRPO

STEERING_METHOD = {
    "category": "structural_control",
    "name": "grpo",
    "control": GRPO,
    "args": GRPOArgs,
}
