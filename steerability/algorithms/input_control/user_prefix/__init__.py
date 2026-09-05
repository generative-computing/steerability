from .args import UserPrefixArgs
from .control import UserPrefix

STEERING_METHOD = {
    "category": "input_control",
    "name": "user_prefix",
    "control": UserPrefix,
    "args": UserPrefixArgs,
}
