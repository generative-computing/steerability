from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.args import SFTArgs
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.base_mixin import SFTTrainerMixin


class SFT(SFTTrainerMixin):
    """
    SFT controller.
    """
    Args = SFTArgs
