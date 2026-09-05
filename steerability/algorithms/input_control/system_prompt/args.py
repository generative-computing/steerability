from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs

_MODES = frozenset({"prepend", "append", "replace"})


@dataclass
class SystemPromptArgs(BaseArgs):
    """Arguments for the system-prompt input control."""

    text: str = field(
        metadata={"help": "System-prompt text set or merged into the leading system message."},
    )

    mode: str = field(
        default="prepend",
        metadata={
            "help": (
                "How the text combines with an existing leading system message: 'prepend' (default), 'append', "
                "or 'replace'. With no existing system message all modes insert the text as a new system message."
            )
        },
    )

    separator: str = field(
        default="\n\n",
        metadata={"help": "String inserted between the text and the existing content for 'prepend'/'append'. "
                          "Empty string allowed."},
    )

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a str; got {type(self.text).__name__}.")
        if not self.text:
            raise ValueError("text must be a non-empty string.")
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}; got {self.mode!r}.")
        if not isinstance(self.separator, str):
            raise TypeError(f"separator must be a str; got {type(self.separator).__name__}.")
