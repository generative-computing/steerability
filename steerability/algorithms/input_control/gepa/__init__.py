from steerability.algorithms.input_control.gepa.args import GEPAArgs
from steerability.algorithms.input_control.gepa.control import GEPA

STEERING_METHOD = {
    "category": "input_control",
    "name": "gepa",
    "control": GEPA,
    "args": GEPAArgs,
}
