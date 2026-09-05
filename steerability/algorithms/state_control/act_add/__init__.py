from .args import ActAddArgs
from .control import ActAdd

STEERING_METHOD = {
    "category": "state_control",
    "name": "act_add",
    "control": ActAdd,
    "args": ActAddArgs,
}
