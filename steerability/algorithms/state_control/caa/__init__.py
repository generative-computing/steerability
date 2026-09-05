from .args import CAAArgs
from .control import CAA

STEERING_METHOD = {
    "category": "state_control",
    "name": "caa",
    "control": CAA,
    "args": CAAArgs,
}
