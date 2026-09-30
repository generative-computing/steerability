"""Linear-AcT registration."""
from .args import LinearAcTArgs
from .control import LinearAcT

STEERING_METHOD = {
    "category": "state_control", "name": "linear_act", "control": LinearAcT, "args": LinearAcTArgs,
}
