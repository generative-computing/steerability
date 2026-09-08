from .args import MergeKitArgs
from .control import MergeKit

# __all__ = ["MergeKit", "MergeKitArgs"]

STEERING_METHOD = {
    "category": "structural_control",
    "name": "mergekit",
    "control": MergeKit,
    "args": MergeKitArgs,
}
