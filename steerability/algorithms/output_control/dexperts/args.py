from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class DExpertsArgs(BaseArgs):
    """Arguments for DExperts (decoding-time experts)."""

    expert_name_or_path: str = field(
        default=None,
        metadata={"help": "HF hub id or local path for the expert LM (steers toward the attribute)."},
    )
    anti_expert_name_or_path: str = field(
        default=None,
        metadata={"help": "HF hub id or local path for the anti-expert LM (steers away from the attribute)."},
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Contrast strength: the expert log-probs enter at +alpha and the anti-expert at -alpha."},
    )
    hf_model_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Extra kwargs passed to AutoModelForCausalLM.from_pretrained() for both auxiliary models."},
    )

    def __post_init__(self) -> None:
        if self.expert_name_or_path is None or self.anti_expert_name_or_path is None:
            raise ValueError("Both 'expert_name_or_path' and 'anti_expert_name_or_path' are required.")
        if not isinstance(self.alpha, (int, float)):
            raise TypeError(f"'alpha' must be a number, got {type(self.alpha).__name__}.")
