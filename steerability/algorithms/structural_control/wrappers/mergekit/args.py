from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class MergeKitArgs(BaseArgs):

    config_path: str | Path | None = None  # yaml config
    config_dict: dict[str, Any] | None = None  # dict config

    out_path: str | Path = "merged-model"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    trust_remote_code: bool = False
    load_merged: bool = True
    force_remerge: bool = False
    device_map: str = "auto"
    allow_cuda: bool = True
    extra_merge_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        if (self.config_path is None) == (self.config_dict is None):
            raise ValueError("Specify either `config_path` or `config_dict`, not both.")

        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"Unsupported dtype '{self.dtype}'.")

        reserved = {"cuda", "trust_remote_code"} & set(self.extra_merge_options)
        if reserved:
            raise ValueError(
                f"`extra_merge_options` may not set {sorted(reserved)}; use `allow_cuda` and `trust_remote_code`."
            )

        self.out_path = Path(self.out_path)  #.expanduser()
        if self.config_path is not None:
            self.config_path = Path(self.config_path)  #.expanduser()
