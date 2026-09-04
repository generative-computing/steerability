from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from peft import PeftType, TaskType

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class TRLArgs(BaseArgs):

    # when the pipeline has no base model of its own, the structural control can load these.
    base_model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    hf_model_kwargs: dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = False

    # datasets / collators
    train_dataset: Any | None = None
    eval_dataset: Any | None = None
    data_collator: Any | None = None

    # training (forwarded to TRL configs)
    training_args: dict[str, Any] = field(default_factory=dict)
    output_dir: str | Path | None = None
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    save_strategy: str = "no"
    load_best_model_at_end: bool = True
    bf16: bool | None = None
    fp16: bool = False
    logging_steps: int = 10
    report_to: str | None = None
    seed: int | None = None
    resume_from_checkpoint: str | None = None

    # PEFT knobs
    use_peft: bool = False
    peft_type: PeftType = PeftType.LORA
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: list[str] | None = None
    lora_dropout: float = 0.05
    bias: str = "none"  # "none" | "all" | "lora_only"
    task_type: TaskType = TaskType.CAUSAL_LM
    init_lora_weights: str | None = None
    use_rslora: bool | None = None
    adapter_name: str | None = "sft"

    # optional in-place LoRA merge after training (no separate control)
    merge_lora_after_train: bool = False
    merged_output_dir: str | Path | None = None  # where to save merged model/tokenizer

    # autofilled PEFT kwargs
    lora_kwargs: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:

        # default transient artifacts under ./tmp so nothing lands at the repository root
        self.output_dir = self.output_dir or self.training_args.get("output_dir") or "./tmp/trainer_output"

        # compose training args
        base_training_args = {
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "load_best_model_at_end": self.load_best_model_at_end,
            "save_strategy": self.save_strategy,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "logging_steps": self.logging_steps,
            "report_to": self.report_to,
            "seed": self.seed,
            "remove_unused_columns": False,
        }
        self.training_args = {**base_training_args, **self.training_args}

        # validation
        if self.r <= 0:
            raise ValueError("LoRA `r` must be > 0.")
        if self.lora_alpha <= 0:
            raise ValueError("`lora_alpha` must be > 0.")
        if not (0 <= self.lora_dropout < 1):
            raise ValueError("`lora_dropout` must be in [0, 1).")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError(f"bias must be 'none'|'all'|'lora_only'; got {self.bias!r}")

        self.lora_kwargs = {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "modules_to_save": self.modules_to_save,
            "task_type": self.task_type,
        }
        if self.init_lora_weights is not None:
            self.lora_kwargs["init_lora_weights"] = self.init_lora_weights
        if self.use_rslora is not None:
            self.lora_kwargs["use_rslora"] = self.use_rslora
