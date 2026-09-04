from aisteer360.algorithms.state_control.common.transforms.context import TransformContext

from .args import ActivationAdapterArgs
from .control import ActivationAdapter

STEERING_METHOD = {
    "category": "state_control",
    "name": "activation_adapter",
    "control": ActivationAdapter,
    "args": ActivationAdapterArgs,
}

__all__ = ["ActivationAdapter", "ActivationAdapterArgs", "TransformContext", "STEERING_METHOD"]
