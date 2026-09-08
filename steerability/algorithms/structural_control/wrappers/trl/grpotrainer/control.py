from steerability.algorithms.structural_control.wrappers.trl.grpotrainer.args import GRPOArgs
from steerability.algorithms.structural_control.wrappers.trl.grpotrainer.base_mixin import GRPOTrainerMixin


class GRPO(GRPOTrainerMixin):
    """
    GRPO controller.
    """
    Args = GRPOArgs
