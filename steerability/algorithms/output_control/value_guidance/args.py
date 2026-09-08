from dataclasses import dataclass, field
from typing import Any

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class ValueGuidanceArgs(BaseArgs):
    """Arguments for the `ValueGuidance` generic.

    A value assigns a per-candidate scalar; the processor shifts the selected candidates' logits by
    `beta * normalize(value)`. Scalar knobs are top-level fields so `ControlSpec` can sweep them;
    the `value` spec is treated as an identity (sweep over whole specs, not inside them).
    """

    value: Any = field(
        default=None,
        metadata={"help": "The candidate value: a BaseCandidateValue instance, a (StepContext) -> "
                          "Tensor[B, K] callable, or a dict spec with a 'kind' key (see resolve_value)."},
    )
    policy: str = field(
        default="top_k",
        metadata={"help": "Candidate policy: 'top_k', 'top_p', or 'surviving'."},
    )
    k: int | None = field(
        default=20,
        metadata={"help": "Candidate count for policy='top_k'."},
    )
    p: float | None = field(
        default=None,
        metadata={"help": "Nucleus threshold in (0, 1] for policy='top_p'."},
    )
    beta: float = field(
        default=1.0,
        metadata={"help": "Shift scale: scores[cand] += beta * normalize(value)."},
    )
    normalize: str = field(
        default="none",
        metadata={"help": "Per-row value normalization: 'none', 'minmax', 'softmax', or 'clamp'."},
    )
    invert: bool = field(
        default=False,
        metadata={"help": "Post-normalization v <- 1 - v (RAD's legacy toxicity head)."},
    )
    mask_non_candidates: bool = field(
        default=True,
        metadata={"help": "Set non-candidate logits to -inf (forced False when policy='surviving')."},
    )
    max_candidates: int | None = field(
        default=None,
        metadata={"help": "Optional clamp on the candidate set (top-N by current score); bounds the "
                          "per-step cost of an unbounded 'surviving' set under a model-forward value."},
    )
    include_in_scoring: bool = field(
        default=True,
        metadata={"help": "Whether this control's processor also applies during compute_logprobs. Set False "
                          "for values too expensive to evaluate per reference position (model-forward)."},
    )

    def __post_init__(self) -> None:
        if self.value is None:
            raise ValueError("'value' is required.")
        if self.policy not in ("top_k", "top_p", "surviving"):
            raise ValueError(f"'policy' must be one of 'top_k', 'top_p', 'surviving', got {self.policy!r}.")
        if self.normalize not in ("none", "minmax", "softmax", "clamp"):
            raise ValueError(
                f"'normalize' must be one of 'none', 'minmax', 'softmax', 'clamp', got {self.normalize!r}."
            )
        if self.policy == "top_k" and (not isinstance(self.k, int) or self.k <= 0):
            raise ValueError(f"policy='top_k' requires a positive 'k', got {self.k!r}.")
        if self.policy == "top_p" and not (self.p is not None and 0.0 < self.p <= 1.0):
            raise ValueError(f"policy='top_p' requires 'p' in (0, 1], got {self.p!r}.")
        if self.max_candidates is not None and (not isinstance(self.max_candidates, int) or self.max_candidates <= 0):
            raise ValueError(f"'max_candidates' must be a positive integer when set, got {self.max_candidates!r}.")
