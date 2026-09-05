"""Inference-Time Intervention (ITI) state control."""
from .args import ITIArgs
from .control import ITI

STEERING_METHOD = {
    "category": "state_control",
    "name": "iti",
    "control": ITI,
    "args": ITIArgs,
}
