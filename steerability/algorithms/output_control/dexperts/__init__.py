from .args import DExpertsArgs
from .control import DExperts

STEERING_METHOD = {
    "category": "output_control",
    "name": "dexperts",
    "control": DExperts,
    "args": DExpertsArgs,
}
