from .args import BudgetForcingArgs
from .control import BudgetForcing

STEERING_METHOD = {
    "category": "output_control",
    "name": "budget_forcing",
    "control": BudgetForcing,
    "args": BudgetForcingArgs,
}
