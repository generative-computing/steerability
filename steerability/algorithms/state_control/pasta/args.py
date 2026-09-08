from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from steerability.algorithms.core.base_args import BaseArgs

if TYPE_CHECKING:
    from steerability.algorithms.state_control.pasta.profiling import HeadProfile


@dataclass
class PASTAArgs(BaseArgs):

    substrings: list[str | list[str]] | None = field(
        default=None,
        metadata={"help": "List of substrings or groups of substrings to steer attention toward or away from."}
    )
    head_config: "dict[int, list[int]] | list[int] | HeadProfile" = field(
        default_factory=lambda: [0, 1],
        metadata={"help": (
            "Either a list of layer indices (to steer all heads), a dict mapping layer index -> list of "
            "head indices, or a HeadProfile recipe resolved at steer() to a dict head map."
        )}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Scaling coefficient controlling the strength of attention modification."}
    )
    scale_position: Literal["include", "exclude", "generation"] = field(
        default="exclude",
        metadata={"help": (
            "'include' upweights the specified tokens, "
            "'exclude' downweights all others, "
            "'generation' applies scaling to the full sequence."
        )}
    )

    # validate
    def __post_init__(self):

        if self.substrings is not None:
            if not isinstance(self.substrings, list):
                raise ValueError("'substrings' must be a list of strings or lists of strings.")
            for item in self.substrings:
                if isinstance(item, str):
                    continue
                if isinstance(item, list):
                    if not all(isinstance(sub, str) for sub in item):
                        raise ValueError("All elements in substring groups must be strings.")
                else:
                    raise ValueError("Each substring must be a string or a list of strings.")

        from steerability.algorithms.state_control.pasta.profiling import HeadProfile

        if isinstance(self.head_config, HeadProfile):
            pass  # a profiling recipe validates its own fields in HeadProfile.__post_init__
        elif isinstance(self.head_config, dict):
            converted: dict[int, list[int]] = {}
            for key, val in self.head_config.items():
                try:
                    layer_idx = int(key)
                except Exception:
                    raise ValueError("All head_config keys must be integers or convertible to integers.")
                if not isinstance(val, list) or not all(isinstance(h, int) for h in val):
                    raise ValueError("head_config values must be lists of integers.")
                converted[layer_idx] = val
            self.head_config = converted
        elif isinstance(self.head_config, list):
            if not all(isinstance(h, int) for h in self.head_config):
                raise ValueError("If head_config is a list, it must contain only integers.")
        else:
            raise ValueError(
                "head_config must be a dict mapping layer->heads, a list of layer indices, or a HeadProfile."
            )

        if not isinstance(self.alpha, (float, int)):
            raise ValueError("alpha must be a float or int.")
        if self.alpha <= 0:
            raise ValueError("alpha must be strictly positive.")

        allowed = {"include", "exclude", "generation"}
        if self.scale_position not in allowed:
            raise ValueError(f"scale_position must be one of {allowed}, got '{self.scale_position}'.")
