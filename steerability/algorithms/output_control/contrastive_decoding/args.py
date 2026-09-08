from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class ContrastiveDecodingArgs(BaseArgs):
    """Arguments for Contrastive Decoding."""

    amateur_name_or_path: str = field(
        default=None,
        metadata={"help": "HF hub id or local path for the amateur (smaller / weaker) LM."},
    )
    alpha: float = field(
        default=0.1,
        metadata={"help": "Plausibility-mask threshold in [0, 1]: keep tokens with p_base(t) >= alpha * max_t p_base(t)."},
    )
    base_weight: float = field(
        default=1.0,
        metadata={"help": "Weight on the base (expert) log-probs."},
    )
    amateur_weight: float = field(
        default=1.0,
        metadata={"help": "Weight subtracted for the amateur log-probs (the amateur enters at -amateur_weight)."},
    )
    hf_model_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Extra kwargs passed to AutoModelForCausalLM.from_pretrained() for the amateur model."},
    )

    def __post_init__(self) -> None:
        if self.amateur_name_or_path is None:
            raise ValueError("'amateur_name_or_path' is required.")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"'alpha' must be in [0, 1], got {self.alpha!r}.")
