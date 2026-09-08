from .args import BestOfNArgs
from .control import BestOfN

STEERING_METHOD = {
    "category": "output_control",
    "name": "best_of_n",
    "control": BestOfN,
    "args": BestOfNArgs,
}
