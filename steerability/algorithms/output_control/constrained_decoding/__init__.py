from .args import ConstrainedDecodingArgs
from .control import ConstrainedDecoding

STEERING_METHOD = {
    "category": "output_control",
    "name": "constrained_decoding",
    "control": ConstrainedDecoding,
    "args": ConstrainedDecodingArgs,
}
