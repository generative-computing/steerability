from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs

_PLACEMENTS = frozenset({"last_user", "first_user", "all_user"})


@dataclass
class UserPrefixArgs(BaseArgs):
    """Arguments for the user-prefix input control."""

    text: str = field(
        metadata={"help": "Marker text prepended to the targeted user turn(s)."},
    )

    separator: str = field(
        default="\n\n",
        metadata={"help": "String inserted between the marker and the existing user content. Empty string allowed."},
    )

    placement: str = field(
        default="last_user",
        metadata={
            "help": (
                "Which user turn(s) receive the marker: 'last_user' (default), 'first_user', or 'all_user'."
            )
        },
    )

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a str; got {type(self.text).__name__}.")
        if not self.text:
            raise ValueError("text must be a non-empty string.")
        if not isinstance(self.separator, str):
            raise TypeError(f"separator must be a str; got {type(self.separator).__name__}.")
        if self.placement not in _PLACEMENTS:
            raise ValueError(f"placement must be one of {sorted(_PLACEMENTS)}; got {self.placement!r}.")
