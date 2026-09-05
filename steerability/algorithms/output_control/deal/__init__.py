from .args import DeALArgs
from .control import DeAL

# __all__ = ["DeAL", "DeALArgs"]

STEERING_METHOD = {
    "category": "output_control",
    "name": "deal",
    "control": DeAL,
    "args": DeALArgs,
}
