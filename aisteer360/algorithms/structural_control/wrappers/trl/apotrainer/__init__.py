from aisteer360.algorithms.structural_control.wrappers.trl.apotrainer.args import APOArgs
from aisteer360.algorithms.structural_control.wrappers.trl.apotrainer.control import APO

# __all__ = ["APO", "APOArgs"]

STEERING_METHOD = {
    "category": "structural_control",
    "name": "apo",
    "control": APO,
    "args": APOArgs,
}
