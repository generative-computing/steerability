from .args import PhasedDecodingArgs
from .control import PhasedDecoding

STEERING_METHOD = {
    "category": "output_control",
    "name": "phased_decoding",
    "control": PhasedDecoding,
    "args": PhasedDecodingArgs,
}
