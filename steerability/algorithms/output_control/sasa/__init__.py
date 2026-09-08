from .args import SASAArgs
from .control import SASA

# __all__ = ["SASA", "SASAArgs"]

STEERING_METHOD = {
    "category": "output_control",
    "name": "sasa",
    "control": SASA,
    "args": SASAArgs,
}
