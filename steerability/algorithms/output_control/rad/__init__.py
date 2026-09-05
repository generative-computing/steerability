from .args import RADArgs
from .control import RAD

# __all__ = ["RAD", "RADArgs"]

STEERING_METHOD = {
    "category": "output_control",
    "name": "rad",
    "control": RAD,
    "args": RADArgs,
}
