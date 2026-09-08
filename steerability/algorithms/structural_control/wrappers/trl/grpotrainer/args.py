from dataclasses import dataclass, field
from typing import Callable

from trl import GRPOConfig

from steerability.algorithms.structural_control.wrappers.trl.args import TRLArgs
from steerability.algorithms.structural_control.wrappers.trl.base_mixin import resolve_config_kwargs


@dataclass
class GRPOArgs(TRLArgs):
    """Arguments for GRPO training via TRL's `GRPOTrainer`.

    GRPO (group-relative policy optimization) is critic-free, so it needs no reward model and no
    value model. The reward is supplied as one or more callables via
    `reward_funcs`; each is called as `reward_func(prompts, completions, **kwargs)` and must return a
    list of floats (one per completion). `num_generations` is the group size used to compute the
    group-relative advantage; `beta` is the KL-to-reference coefficient (set `beta=0.0` to disable the
    reference model entirely).

    `GRPOTrainer` reads a text `"prompt"` column directly with no pre-tokenization and uses each
    prompt as provided, so prompts are pre-truncated in the dataset when a cap is needed.
    `num_generations` must be >= 2 and evenly divide the global train batch size. The convenience
    fields supply defaults; an entry of the same name in `training_args` overrides the field.
    """

    reward_funcs: Callable | list[Callable] | None = field(
        default=None,
        metadata={
            "help": (
                "Reward callable, or list of callables, forwarded to TRL's `GRPOTrainer`. Each has "
                "signature `reward_func(prompts, completions, **kwargs) -> list[float]`."
            )
        },
    )

    num_generations: int = field(
        default=8,
        metadata={"help": "Group size G (>= 2). Must evenly divide the global train batch size."},
    )
    max_completion_length: int = field(
        default=64,
        metadata={"help": "Max generated tokens per completion during rollouts."},
    )
    temperature: float = field(
        default=0.9,
        metadata={"help": "Sampling temperature for rollouts."},
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "KL-to-reference coefficient. 0.0 disables the reference model."},
    )

    learning_rate: float = field(default=1e-6, metadata={"help": "GRPO uses a small LR (matches TRL default)."})
    load_best_model_at_end: bool = False  # GRPO has no eval split by default

    def __post_init__(self) -> None:
        super().__post_init__()

        funcs = self.reward_funcs
        if funcs is None or (isinstance(funcs, list) and len(funcs) == 0):
            raise ValueError("`GRPOArgs` requires at least one entry in `reward_funcs`.")

        if self.num_generations < 2:
            raise ValueError(f"`num_generations` must be >= 2 for GRPO; got {self.num_generations}.")

        # GRPOTrainer requires num_generations to evenly divide the global train batch size
        # (per_device_train_batch_size * num_processes). Validate the single-process case here as an
        # early, friendly check; the trainer performs the authoritative multi-process check.
        if self.per_device_train_batch_size % self.num_generations != 0:
            raise ValueError(
                f"`per_device_train_batch_size` ({self.per_device_train_batch_size}) must be evenly "
                f"divisible by `num_generations` ({self.num_generations}) for GRPO. Adjust either value "
                "so the (single-process) global train batch size is a multiple of num_generations."
            )

        for key in ("num_generations", "max_completion_length", "temperature", "beta"):
            self.training_args.setdefault(key, getattr(self, key))

        resolve_config_kwargs(GRPOConfig, self.training_args)
