"""S-space registration."""
from .args import SSpaceArgs
from .control import SSpace

STEERING_METHOD = {
    "category": "state_control", "name": "sspace", "control": SSpace, "args": SSpaceArgs,
}
