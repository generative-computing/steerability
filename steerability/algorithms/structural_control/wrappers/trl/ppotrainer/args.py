import warnings
from dataclasses import dataclass, field

from steerability.algorithms.structural_control.wrappers.trl.args import TRLArgs


@dataclass
class PPOArgs(TRLArgs):
    """Arguments for PPO training via TRL's `PPOTrainer`.

    Reward and value models are sequence-classification models (`AutoModelForSequenceClassification`).
    `value_model_name_or_path` defaults to `reward_model_name_or_path`; the wrapper materializes a
    separate copy at training time because TRL's `PPOTrainer` wraps the value model into its policy
    and cannot accept `value_model=None`.
    """

    reward_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "HF id or local path of a sequence-classification reward model."},
    )
    value_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional separate value model. If None, a fresh copy of the reward model is loaded "
                "and used as the value head."
            )
        },
    )

    kl_coef: float = field(default=0.05, metadata={"help": "KL penalty coefficient."})
    num_ppo_epochs: int = field(default=4, metadata={"help": "PPO update epochs per rollout batch."})
    temperature: float = field(default=0.7, metadata={"help": "Sampling temperature for rollouts."})
    response_length: int = field(default=53, metadata={"help": "Max response tokens per rollout."})
    local_rollout_forward_batch_size: int = field(
        default=64, metadata={"help": "Forward batch size during rollout generation."}
    )
    missing_eos_penalty: float | None = field(
        default=None,
        metadata={"help": "Penalty for rollouts that don't end in EOS. None disables."},
    )

    learning_rate: float = field(default=3e-6, metadata={"help": "PPO defaults to a smaller LR than DPO."})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reward_model_name_or_path is None:
            raise ValueError(
                "`reward_model_name_or_path` is required for PPO. Pass an HF "
                "AutoModelForSequenceClassification id or local path."
            )
        for key in (
            "kl_coef",
            "num_ppo_epochs",
            "temperature",
            "response_length",
            "local_rollout_forward_batch_size",
        ):
            self.training_args.setdefault(key, getattr(self, key))
        if self.missing_eos_penalty is not None:
            self.training_args.setdefault("missing_eos_penalty", self.missing_eos_penalty)

        # PPOConfig lives under trl.experimental; validate at construction when it is importable
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from trl.experimental.ppo import PPOConfig
        except ImportError:
            return
        from steerability.algorithms.structural_control.wrappers.trl.base_mixin import resolve_config_kwargs

        resolve_config_kwargs(PPOConfig, self.training_args)
