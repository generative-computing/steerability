"""CorDA-derived PCA registration."""
from .args import CordaPCAArgs
from .control import CordaPCA

STEERING_METHOD = {
    "category": "state_control", "name": "corda_pca", "control": CordaPCA, "args": CordaPCAArgs,
}
