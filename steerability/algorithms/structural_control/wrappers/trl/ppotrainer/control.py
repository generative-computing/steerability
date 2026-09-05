from steerability.algorithms.structural_control.wrappers.trl.ppotrainer.args import PPOArgs
from steerability.algorithms.structural_control.wrappers.trl.ppotrainer.base_mixin import PPOTrainerMixin


class PPO(PPOTrainerMixin):
    """
    PPO controller.
    """
    Args = PPOArgs
