from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class StoppingRulesArgs(BaseArgs):
    """Arguments for the `StoppingRules` generic.

    Configure any of three per-row stops; at least one must be set.
    """

    stop_texts: list = field(
        default_factory=list,
        metadata={"help": "Substrings that halt a row once its continuation contains one of them."},
    )
    stop_token_ids: list = field(
        default_factory=list,
        metadata={"help": "Token ids that halt a row once its last token is one of them."},
    )
    budget: int | None = field(
        default=None,
        metadata={"help": "Max new tokens (past the prompt) before a row halts. None disables."},
    )

    def __post_init__(self) -> None:
        if not self.stop_texts and not self.stop_token_ids and self.budget is None:
            raise ValueError("Configure at least one of 'stop_texts', 'stop_token_ids', or 'budget'.")
        if self.budget is not None and (not isinstance(self.budget, int) or self.budget <= 0):
            raise ValueError(f"'budget' must be a positive integer when set, got {self.budget!r}.")
