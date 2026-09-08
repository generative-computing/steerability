from .args import SearchDecodingArgs
from .control import SearchDecoding

STEERING_METHOD = {
    "category": "output_control",
    "name": "search_decoding",
    "control": SearchDecoding,
    "args": SearchDecodingArgs,
}
