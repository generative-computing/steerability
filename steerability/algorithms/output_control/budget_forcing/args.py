from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class BudgetForcingArgs(BaseArgs):
    """Arguments for Budget Forcing (s1 test-time reasoning-length control)."""

    max_thinking_tokens: int = field(
        default=512,
        metadata={"help": "Token budget for each thinking segment before the closing think tag is forced."},
    )
    extension_text: str = field(
        default="Wait",
        metadata={"help": "Text appended to prolong reasoning when the model tries to stop early."},
    )
    num_extensions: int = field(
        default=0,
        metadata={"help": "Number of times to append `extension_text` and continue thinking (0 disables extension)."},
    )
    end_think: str = field(
        default="</think>",
        metadata={"help": "The closing-think marker: both the thinking-phase boundary and the forced tokens before the answer."},
    )
    end_think_token_ids: tuple[int, ...] = field(
        default=(),
        metadata={"help": "Token ids that also end each thinking phase (alongside `end_think`); the backend-portable "
                          "form for a closing-think delimiter that tokenizes to a special token."},
    )

    def __post_init__(self) -> None:
        if not isinstance(self.max_thinking_tokens, int) or self.max_thinking_tokens <= 0:
            raise ValueError(f"'max_thinking_tokens' must be a positive integer, got {self.max_thinking_tokens!r}.")
        if not isinstance(self.num_extensions, int) or self.num_extensions < 0:
            raise ValueError(f"'num_extensions' must be a non-negative integer, got {self.num_extensions!r}.")
        if not self.end_think:
            raise ValueError("'end_think' must be a non-empty string.")
        if isinstance(self.end_think_token_ids, (str, bytes)):
            raise ValueError("'end_think_token_ids' must be a sequence of ints, not a string.")
        self.end_think_token_ids = tuple(int(i) for i in self.end_think_token_ids)
