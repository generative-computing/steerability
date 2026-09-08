from .args import StoppingRulesArgs
from .control import StoppingRules

STEERING_METHOD = {
    "category": "output_control",
    "name": "stopping_rules",
    "control": StoppingRules,
    "args": StoppingRulesArgs,
}
