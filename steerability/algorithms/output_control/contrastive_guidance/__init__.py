from .args import ContrastiveGuidanceArgs
from .control import ContrastiveGuidance

STEERING_METHOD = {
    "category": "output_control",
    "name": "contrastive_guidance",
    "control": ContrastiveGuidance,
    "args": ContrastiveGuidanceArgs,
}
