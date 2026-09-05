from .args import FewShotArgs
from .control import FewShot

# __all__ = ["FewShot", "FewShotArgs"]

STEERING_METHOD = {
    "category": "input_control",
    "name": "few_shot",
    "control": FewShot,
    "args": FewShotArgs,
}
