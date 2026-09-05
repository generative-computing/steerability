from typing import Any

from peft import LoraConfig, PeftType
from transformers import DataCollatorForLanguageModeling, PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from steerability.algorithms.structural_control.base import StructuralControl
from steerability.algorithms.structural_control.wrappers.trl.base_mixin import TRLMixin, resolve_config_kwargs


class SFTTrainerMixin(TRLMixin, StructuralControl):
    """
    SFT structural control (backed by TRL's SFTTrainer).
    """

    # filled by Args dataclass
    train_dataset: Any | None = None
    eval_dataset: Any | None = None
    data_collator: Any | None = None

    def steer(self, model: PreTrainedModel | None, tokenizer: PreTrainedTokenizerBase | None = None, **_) -> PreTrainedModel:

        self.tokenizer = tokenizer or (getattr(model, "tokenizer", None) if model is not None else None)

        # resolve or load as needed
        model, self.tokenizer = self._resolve_model_tokenizer(model, self.tokenizer)

        # build TRL config
        config_kwargs = resolve_config_kwargs(SFTConfig, self.training_args)
        training_config = SFTConfig(**config_kwargs)

        # build PEFT config
        peft_config = None
        if self.use_peft and self.peft_type == PeftType.LORA:
            peft_config = LoraConfig(**self.lora_kwargs)

        # default collator for tokenized datasets (labels from input_ids)
        data_collator = self.data_collator
        if data_collator is None and self.train_dataset is not None and hasattr(self.train_dataset, "features"):
            if "labels" not in self.train_dataset.features and "input_ids" in self.train_dataset.features:
                data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # train if a dataset is provided
        if self.train_dataset is not None:
            trainer = SFTTrainer(
                model=model,
                args=training_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=data_collator,
                processing_class=self.tokenizer,
                peft_config=peft_config,
            )
            trainer.train(resume_from_checkpoint=self.training_args.get("resume_from_checkpoint"))
            model = trainer.model
            self._maybe_save_trained_artifacts(trainer)

            # optional in-place LoRA merge, then re-freeze
            model = self._maybe_merge_lora_in_place(model)

        return self._post_train_freeze(model)
