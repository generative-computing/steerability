import os
from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class SASAArgs(BaseArgs):
    """Arguments for `SASA` (subspace-margin guided sampling)."""

    beta: float = field(
        default=0.0,
        metadata={"help": "Scaling coefficient for value redistribution."},
    )
    wv_path: str | None = field(
        default=None,
        metadata={"help": "Path to a saved probe: a probe directory (safetensors plus JSON sidecar), a "
                          "`.probe` JSON file, or a legacy `.pt` tensor checkpoint."},
    )
    gen_wv_data_path: str | None = field(
        default="Jigsaw_data/",
        metadata={"help": "Path to the labeled attribute dataset used to fit the probe (defaults to the "
                          "Jigsaw toxicity corpus layout)."},
    )
    gen_wv_data: dict | None = field(
        default=None,
        metadata={"help": "In-memory labeled data as `{'pos': [...], 'neg': [...]}` (e.g. non-toxic/toxic "
                          "sentences)."},
    )
    gen_wv_length: int | None = field(
        default=-1,
        metadata={"help": "The maximum number of samples used for preparing SASA steering if wv_path does not exist."}
    )
    gen_wv_batch_size: int | None = field(
        default=4,
        metadata={"help": "The batch size used for preparing SASA steering if wv_path does not exist."}
    )
    max_candidates: int | None = field(
        default=None,
        metadata={"help": "Optional clamp on the surviving candidate set (top-N by score) to bound the "
                          "per-step model forward. None (default) leaves the surviving set unbounded."}
    )

    # validation
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError("'beta' must be non-negative.")
        if self.wv_path is not None and not (
            os.path.isdir(self.wv_path) or self.wv_path.endswith((".pt", ".probe"))
        ):
            raise ValueError(
                "wv_path must be a probe directory, a .pt tensor checkpoint, or a .probe JSON file."
            )
        if self.wv_path is None and self.gen_wv_batch_size < 0:
            raise ValueError("'gen_wv_batch_size' must be non-negative.")
