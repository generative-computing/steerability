from .args import VJPDeltaArgs
from .control import VJPDelta
from .fit import VJPDeltaFit

STEERING_METHOD = {
    "category": "state_control",
    "name": "vjp_delta",
    "control": VJPDelta,
    "args": VJPDeltaArgs,
}

__all__ = ["VJPDelta", "VJPDeltaArgs", "VJPDeltaFit"]
