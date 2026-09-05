from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class RADArgs(BaseArgs):
    """Arguments for RAD (Reward-Augmented Decoding)."""

    reward_model_id: str = field(
        metadata={"help": "HF model id or local path for an AutoModelForSequenceClassification reward model."},
    )
    beta: float = field(
        metadata={"help": "Steering intensity (Algorithm 1's beta). Non-negative; direction is set by 'invert'."},
    )
    top_k: int = field(
        default=20,
        metadata={"help": "Number of candidate tokens scored per step (Algorithm 1's k)."},
    )
    invert: bool = field(
        default=False,
        metadata={"help": "Use 1 - reward as the shift (steer away from the scored attribute)."},
    )
    score_index: int = field(
        default=0,
        metadata={"help": "Output column of the reward model read as the score."},
    )
    score_transform: str = field(
        default="none",
        metadata={"help": "Map head outputs to [0, 1]: 'none', 'sigmoid', or 'softmax' (softmax over all "
                          "columns, then select score_index)."},
    )
    reward_model_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Extra kwargs for AutoModelForSequenceClassification.from_pretrained()."},
    )
    include_in_scoring: bool = field(
        default=True,
        metadata={"help": "Apply the processor during compute_logprobs (one aux forward per reference position)."},
    )
    efficient: bool = field(
        default=True,
        metadata={"help": "Cache reward-model prefix activations across steps when preconditions hold "
                          "(unidirectional reward model sharing the LM's vocabulary)."},
    )

    def __post_init__(self):
        if not self.reward_model_id:
            raise ValueError("'reward_model_id' must be a non-empty model id or path.")
        if self.beta < 0:
            raise ValueError("'beta' must be non-negative.")
        if self.top_k < 1:
            raise ValueError("'top_k' must be at least 1.")
        if self.score_index < 0:
            raise ValueError("'score_index' must be non-negative.")
        if self.score_transform not in ("none", "sigmoid", "softmax"):
            raise ValueError("'score_transform' must be one of 'none', 'sigmoid', 'softmax'.")
