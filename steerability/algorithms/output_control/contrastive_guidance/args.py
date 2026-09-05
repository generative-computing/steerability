from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class ContrastiveGuidanceArgs(BaseArgs):
    """Arguments for the `ContrastiveGuidance` generic.

    Mixes the base log-probs with weighted auxiliary log-prob sources as
    `w0 * log p_base + sum_i w_i * log p_source_i`. `sources` and `weights` are parallel top-level
    lists precisely so `ControlSpec` can sweep them (e.g. `vars={"weights": [[1,-1], [2,-2]]}`).
    """

    sources: list = field(
        default=None,
        metadata={"help": "List of source specs (each a BaseLogitSource instance, a callable, a str "
                          "aux-model name/path, or a dict spec with a 'kind' key)."},
    )
    weights: list = field(
        default=None,
        metadata={"help": "Weights parallel to 'sources' (source i enters at weight w_i)."},
    )
    base_weight: float = field(
        default=1.0,
        metadata={"help": "Weight w0 on the base model's log-probs."},
    )
    alpha: float | None = field(
        default=None,
        metadata={"help": "Plausibility-mask threshold in (0, 1]; keep tokens with p_base(t) >= "
                          "alpha * max_t p_base(t). None disables the mask."},
    )
    include_in_scoring: bool = field(
        default=True,
        metadata={"help": "Whether this control's processor also applies during compute_logprobs."},
    )

    def __post_init__(self) -> None:
        if self.sources is None or self.weights is None:
            raise ValueError("Both 'sources' and 'weights' are required.")
        if len(self.sources) == 0:
            raise ValueError("'sources' must be non-empty.")
        if len(self.sources) != len(self.weights):
            raise ValueError(
                f"'sources' ({len(self.sources)}) and 'weights' ({len(self.weights)}) must be equal length."
            )
        if self.alpha is not None and not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"'alpha' must be in (0, 1] when set, got {self.alpha!r}.")
