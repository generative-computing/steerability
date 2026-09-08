from .args import CASTArgs
from .control import CAST

# __all__ = ["CAST", "CASTArgs"]

STEERING_METHOD = {
    "category": "state_control",
    "name": "cast",
    "control": CAST,
    "args": CASTArgs,
}
