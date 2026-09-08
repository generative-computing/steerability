from .args import DirectionalAblationArgs
from .control import DirectionalAblation

STEERING_METHOD = {
    "category": "state_control",
    "name": "directional_ablation",
    "control": DirectionalAblation,
    "args": DirectionalAblationArgs,
}
