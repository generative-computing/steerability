from .args import ValueGuidanceArgs
from .control import ValueGuidance

STEERING_METHOD = {
    "category": "output_control",
    "name": "value_guidance",
    "control": ValueGuidance,
    "args": ValueGuidanceArgs,
}
