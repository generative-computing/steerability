from steerability.algorithms.input_control.prewrite.args import PRewriteArgs
from steerability.algorithms.input_control.prewrite.control import PRewrite

STEERING_METHOD = {
    "category": "input_control",
    "name": "prewrite",
    "control": PRewrite,
    "args": PRewriteArgs,
}
