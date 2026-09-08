from dataclasses import dataclass, field

from trl import DPOConfig

from steerability.algorithms.structural_control.wrappers.trl.args import TRLArgs
from steerability.algorithms.structural_control.wrappers.trl.base_mixin import resolve_config_kwargs
from steerability.utils.rendering import PromptFormat


@dataclass
class DPOArgs(TRLArgs):
    """Arguments for DPO training via TRL's `DPOTrainer`.

    `loss_type` is a single loss name, or a list of names combined with `loss_weights`. The list
    form `["sigmoid", "sft"]` with `loss_weights=[1.0, alpha]` adds a negative log-likelihood term
    on the chosen completion (weight `alpha`) alongside the sigmoid preference loss, which keeps the
    chosen completion's likelihood from falling when chosen and rejected completions are
    near-identical. The convenience fields (`beta`, `loss_type`, `loss_weights`, `max_length`,
    `learning_rate`, ...) supply defaults; an entry of the same name in `training_args` overrides the
    field. Sequence truncation is by `max_length`.
    """

    loss_type: str | list[str] = field(default="sigmoid")
    loss_weights: list[float] | None = field(default=None)
    beta: float = field(default=0.1)
    learning_rate: float = field(default=1e-6)
    max_length: int | None = field(default=1024)
    prompt_format: PromptFormat = field(default="raw")

    # optional
    precompute_ref_log_probs: bool | None = True
    disable_dropout: bool | None = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prompt_format not in ("raw", "chat_completion", "chat_prompt"):
            raise ValueError(
                f"prompt_format must be 'raw', 'chat_completion', or 'chat_prompt'; got {self.prompt_format!r}."
            )
        loss_types = [self.loss_type] if isinstance(self.loss_type, str) else list(self.loss_type)
        if not loss_types or not all(isinstance(name, str) and name for name in loss_types):
            raise ValueError("loss_type must be a non-empty loss name or a list of loss names.")
        if self.loss_weights is not None and len(self.loss_weights) != len(loss_types):
            raise ValueError(
                f"loss_weights must have one weight per loss_type entry; got {len(self.loss_weights)} "
                f"weights for {len(loss_types)} loss types."
            )

        # convenience fields are defaults; an explicit training_args entry wins
        self.training_args.setdefault("beta", self.beta)
        self.training_args.setdefault("loss_type", self.loss_type)
        if self.loss_weights is not None:
            self.training_args.setdefault("loss_weights", list(self.loss_weights))
        self.training_args.setdefault("max_length", self.max_length)
        if self.precompute_ref_log_probs is not None:
            self.training_args.setdefault("precompute_ref_log_probs", self.precompute_ref_log_probs)
        if self.disable_dropout is not None:
            self.training_args.setdefault("disable_dropout", self.disable_dropout)

        # fail at construction, before any model is loaded
        resolve_config_kwargs(DPOConfig, self.training_args)
