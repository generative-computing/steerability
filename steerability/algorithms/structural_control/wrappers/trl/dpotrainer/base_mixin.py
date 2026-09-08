import torch
from peft import LoraConfig, PeftType
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from steerability.algorithms.structural_control.base import StructuralControl
from steerability.algorithms.structural_control.wrappers.trl.base_mixin import TRLMixin, resolve_config_kwargs
from steerability.algorithms.structural_control.wrappers.trl.utils.preference_schema import (
    standardize_preference_dataset,
)
from steerability.utils.rendering import PromptFormat


class DPOTrainerMixin(TRLMixin, StructuralControl):
    """DPO structural control backed by TRL's `DPOTrainer`.

    Preference rows are normalized to plain-string `prompt`/`chosen`/`rejected` columns before
    training. With the default `prompt_format="raw"`, TRL trains on the strings verbatim. A
    non-`"raw"` `prompt_format` renders each prompt through the tokenizer's chat template with
    the assistant generation prompt appended, matching how the pipeline renders prompts at
    inference, while the completions stay bare strings; this requires a chat-templated tokenizer.

    `loss_type` is a single loss name or a list of names combined with `loss_weights`. The list form
    `loss_type=["sigmoid", "sft"]` with `loss_weights=[1.0, alpha]` adds a negative log-likelihood
    term on the chosen completion (weight `alpha`) alongside the sigmoid preference loss, which
    keeps the chosen completion's likelihood from falling when chosen and rejected completions are
    near-identical. The convenience fields (`beta`, `loss_type`, `loss_weights`, `max_length`, ...)
    supply defaults; an entry of the same name in `training_args` overrides the field. `training_args`
    is forwarded verbatim to `DPOConfig`, and a key it does not declare raises.
    """

    train_dataset = None
    eval_dataset = None
    ref_model: PreTrainedModel | None = None
    prompt_format: PromptFormat = "raw"

    # optional
    precompute_ref_log_probs: bool | None = True
    disable_dropout: bool | None = True

    def steer(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        ref_model: PreTrainedModel | None = None,
        **_,
    ) -> torch.nn.Module:

        self.tokenizer = tokenizer or (getattr(model, "tokenizer", None) if model is not None else None)

        # resolve or load model/tokenizer
        model, self.tokenizer = self._resolve_model_tokenizer(model, self.tokenizer)

        # clean
        if self.train_dataset is not None:
            self.train_dataset = standardize_preference_dataset(
                self.train_dataset, prompt_format=self.prompt_format, tokenizer=self.tokenizer,
            )
        if self.eval_dataset is not None:
            self.eval_dataset = standardize_preference_dataset(
                self.eval_dataset, prompt_format=self.prompt_format, tokenizer=self.tokenizer,
            )

        # compose config kwargs (optional DPO fields); an explicit training_args entry wins
        config_kwargs = dict(self.training_args)
        if self.precompute_ref_log_probs is not None:
            config_kwargs.setdefault("precompute_ref_log_probs", self.precompute_ref_log_probs)
        if self.disable_dropout is not None:
            config_kwargs.setdefault("disable_dropout", self.disable_dropout)

        config_kwargs = resolve_config_kwargs(DPOConfig, config_kwargs)
        training_config = DPOConfig(**config_kwargs)

        # build PEFT config
        peft_config = None
        if self.use_peft and self.peft_type == PeftType.LORA:
            peft_config = LoraConfig(**self.lora_kwargs)
            ref_model = None  # TRL constructs frozen ref from base weights

        # train if a dataset is provided
        if self.train_dataset is not None:
            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
                peft_config=peft_config,
            )
            trainer.train(resume_from_checkpoint=self.training_args.get("resume_from_checkpoint"))
            model = trainer.model
            self._maybe_save_trained_artifacts(trainer)
            model = self._maybe_merge_lora_in_place(model)

        return self._post_train_freeze(model)
