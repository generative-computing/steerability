from steerability.algorithms.input_control.cpo.args import CPOArgs
from steerability.algorithms.input_control.cpo.control import CPO

STEERING_METHOD = {
    "category": "input_control",
    "name": "cpo",
    "control": CPO,
    "args": CPOArgs,
}
