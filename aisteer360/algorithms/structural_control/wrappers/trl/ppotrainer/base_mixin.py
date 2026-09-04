from typing import Any

import torch
from peft import LoraConfig, PeftType
from transformers import AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer
from trl import PPOConfig, PPOTrainer

from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.algorithms.structural_control.wrappers.trl.base_mixin import TRLMixin
from aisteer360.algorithms.structural_control.wrappers.trl.utils.prompt_schema import standardize_prompt_dataset
from aisteer360.utils.tokenization import ensure_pad_token


class PPOTrainerMixin(TRLMixin, StructuralControl):
    """PPO structural control backed by TRL's `PPOTrainer`.

    Reward and value models are sequence-classification models. When
    `value_model_name_or_path` is not provided, the wrapper loads a fresh value model from the same
    path as the reward model, since TRL's `PPOTrainer` wraps the value model into its policy and
    rejects `value_model=None`.

    The standardized prompt-only training dataset is tokenized into an `input_ids` column before
    being passed to the trainer, which reads `data["input_ids"]` directly.

    Note: reward and value models must share the policy's tokenizer/vocabulary. The wrapper guards
    this with an error.
    """

    train_dataset: Any | None = None
    eval_dataset: Any | None = None
    reward_model_name_or_path: str | None = None
    value_model_name_or_path: str | None = None
    max_prompt_length: int = 512

    def steer(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None = None,
        ref_model: PreTrainedModel | None = None,
        **_,
    ) -> torch.nn.Module:
        self.tokenizer = tokenizer or (getattr(model, "tokenizer", None) if model is not None else None)
        model, self.tokenizer = self._resolve_model_tokenizer(model, self.tokenizer)
        self.tokenizer = ensure_pad_token(self.tokenizer)

        # reward + value models (sequence-classification heads)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_name_or_path,
            num_labels=1,
            trust_remote_code=self.trust_remote_code,
        )
        value_path = self.value_model_name_or_path or self.reward_model_name_or_path
        value_model = AutoModelForSequenceClassification.from_pretrained(
            value_path,
            num_labels=1,
            trust_remote_code=self.trust_remote_code,
        )

        self._check_scoring_vocab(reward_model, value_model)

        # standardize then tokenize so the dataset has the input_ids column PPOTrainer expects
        train_dataset = self._prepare_dataset(self.train_dataset) if self.train_dataset is not None else None
        eval_dataset = self._prepare_dataset(self.eval_dataset) if self.eval_dataset is not None else None

        config_kwargs = self._filter_kwargs_for_class_or_callable(PPOConfig, self.training_args)
        training_config = PPOConfig(**config_kwargs)

        peft_config = None
        if self.use_peft and self.peft_type == PeftType.LORA:
            peft_config = LoraConfig(**self.lora_kwargs)
            ref_model = None

        if train_dataset is not None:
            trainer = PPOTrainer(
                args=training_config,
                processing_class=self.tokenizer,
                model=model,
                ref_model=ref_model,
                reward_model=reward_model,
                value_model=value_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
            )
            trainer.train()

            # recover the trained policy so it can be used for generation
            trained_model = trainer.accelerator.unwrap_model(trainer.model)
            model = getattr(trained_model, "policy", trained_model)
            self._maybe_save_trained_artifacts(trainer)
            model = self._maybe_merge_lora_in_place(model)

        return self._post_train_freeze(model)

    def _check_scoring_vocab(self, reward_model, value_model) -> None:
        """Verify the reward/value models can index every policy token id.
        """
        policy_vocab = len(self.tokenizer)
        for role, scoring_model, path in (
            ("reward", reward_model, self.reward_model_name_or_path),
            ("value", value_model, self.value_model_name_or_path or self.reward_model_name_or_path),
        ):
            scoring_vocab = getattr(scoring_model.config, "vocab_size", None)
            if scoring_vocab is not None and scoring_vocab < policy_vocab:
                raise ValueError(
                    f"PPO {role} model '{path}' has vocab_size {scoring_vocab}, smaller than the "
                    f"policy tokenizer's {policy_vocab}. TRL's PPOTrainer scores the policy's own token "
                    f"ids with one shared tokenizer, so the {role} model must share the policy's "
                    "tokenizer/vocabulary (use a reward/value model from the policy's model family)."
                )

    def _prepare_dataset(self, dataset):
        """Standardize a prompt-bearing dataset and tokenize prompts into the `input_ids` column TRL expects."""
        ds = standardize_prompt_dataset(dataset)
        max_length = self.max_prompt_length

        def tokenize(example):
            return self.tokenizer(
                example["prompt"],
                truncation=True,
                max_length=max_length,
            )

        return ds.map(tokenize, remove_columns=["prompt"])
