from .args import ContrastiveDecodingArgs
from .control import ContrastiveDecoding

STEERING_METHOD = {
    "category": "output_control",
    "name": "contrastive_decoding",
    "control": ContrastiveDecoding,
    "args": ContrastiveDecodingArgs,
}
