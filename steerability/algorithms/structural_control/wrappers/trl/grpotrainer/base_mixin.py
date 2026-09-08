from typing import Any, Callable

import torch
from peft import LoraConfig, PeftType
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

from steerability.algorithms.structural_control.base import StructuralControl
from steerability.algorithms.structural_control.wrappers.trl.base_mixin import TRLMixin, resolve_config_kwargs
from steerability.algorithms.structural_control.wrappers.trl.utils.prompt_schema import standardize_prompt_dataset
from steerability.utils.tokenization import ensure_pad_token


class GRPOTrainerMixin(TRLMixin, StructuralControl):
    """GRPO structural control backed by TRL's `GRPOTrainer`.

    GRPO is critic-free (group-relative advantage), so there is no reward model and no value model.
    The reward comes from one or more callables in `reward_funcs`, each called as
    `reward_func(prompts, completions, **kwargs)` and returning one float per completion. The trainer
    reads a text `"prompt"` column directly and samples `num_generations` completions per prompt, so
    the dataset is only standardized to a prompt column and is not tokenized into an `input_ids`
    column.
    """

    train_dataset: Any | None = None
    eval_dataset: Any | None = None
    reward_funcs: Callable | list[Callable] | None = None

    def steer(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **_,
    ) -> torch.nn.Module:
        self.tokenizer = tokenizer or (getattr(model, "tokenizer", None) if model is not None else None)
        model, self.tokenizer = self._resolve_model_tokenizer(model, self.tokenizer)
        self.tokenizer = ensure_pad_token(self.tokenizer)

        if self.reward_funcs is None:
            raise ValueError(
                "GRPOTrainerMixin.steer: `reward_funcs` is required (a callable or list of callables)."
            )
        reward_funcs = self.reward_funcs if isinstance(self.reward_funcs, list) else [self.reward_funcs]

        # GRPO reads the text "prompt" column directly; standardize but do NOT tokenize to input_ids
        train_dataset = standardize_prompt_dataset(self.train_dataset) if self.train_dataset is not None else None
        eval_dataset = standardize_prompt_dataset(self.eval_dataset) if self.eval_dataset is not None else None

        config_kwargs = resolve_config_kwargs(GRPOConfig, self.training_args)
        training_config = GRPOConfig(**config_kwargs)

        peft_config = None
        if self.use_peft and self.peft_type == PeftType.LORA:
            peft_config = LoraConfig(**self.lora_kwargs)

        if train_dataset is not None:
            trainer = GRPOTrainer(
                model=model,
                reward_funcs=reward_funcs,
                args=training_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=self.tokenizer,
                peft_config=peft_config,
            )
            trainer.train(resume_from_checkpoint=self.training_args.get("resume_from_checkpoint"))

            # recover the trained policy so it can be used for generation (GRPO has no .policy wrapper)
            trained_model = trainer.accelerator.unwrap_model(trainer.model)
            model = getattr(trained_model, "policy", trained_model)
            self._maybe_save_trained_artifacts(trainer)
            model = self._maybe_merge_lora_in_place(model)

        return self._post_train_freeze(model)
