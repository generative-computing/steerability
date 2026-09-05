from .args import PASTAArgs
from .control import PASTA
from .profiling import HeadProfile, HeadProfileResult

STEERING_METHOD = {
    "category": "state_control",
    "name": "pasta",
    "control": PASTA,
    "args": PASTAArgs,
}
