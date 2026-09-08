from .args import AngularSteeringArgs
from .control import AngularSteering

STEERING_METHOD = {
    "category": "state_control",
    "name": "angular_steering",
    "control": AngularSteering,
    "args": AngularSteeringArgs,
}
