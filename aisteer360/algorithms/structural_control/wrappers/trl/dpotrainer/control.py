from aisteer360.algorithms.structural_control.wrappers.trl.dpotrainer.args import DPOArgs
from aisteer360.algorithms.structural_control.wrappers.trl.dpotrainer.base_mixin import DPOTrainerMixin


class DPO(DPOTrainerMixin):
    """
    DPO controller.
    """
    Args = DPOArgs
