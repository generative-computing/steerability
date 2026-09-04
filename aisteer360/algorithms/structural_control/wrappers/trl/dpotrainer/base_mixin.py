import torch
from peft import LoraConfig, PeftType
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import DPOConfig, DPOTrainer

from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.algorithms.structural_control.wrappers.trl.base_mixin import TRLMixin
from aisteer360.algorithms.structural_control.wrappers.trl.utils.preference_schema import standardize_preference_dataset


class DPOTrainerMixin(TRLMixin, StructuralControl):
    """
    DPO structural control backed by TRL's DPOTrainer.
    """

    train_dataset = None
    eval_dataset = None
    ref_model: PreTrainedModel | None = None

    # optional
    precompute_ref_log_probs: bool | None = True
    disable_dropout: bool | None = True

    def steer(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None = None,
        ref_model: PreTrainedModel | None = None,
        **_,
    ) -> torch.nn.Module:

        self.tokenizer = tokenizer or (getattr(model, "tokenizer", None) if model is not None else None)

        # resolve or load model/tokenizer
        model, self.tokenizer = self._resolve_model_tokenizer(model, self.tokenizer)

        # clean
        if self.train_dataset is not None:
            self.train_dataset = standardize_preference_dataset(self.train_dataset)
        if self.eval_dataset is not None:
            self.eval_dataset = standardize_preference_dataset(self.eval_dataset)

        # compose config kwargs (optional DPO fields)
        config_kwargs = dict(self.training_args)
        if self.precompute_ref_log_probs is not None:
            config_kwargs["precompute_ref_log_probs"] = self.precompute_ref_log_probs
        if self.disable_dropout is not None:
            config_kwargs["disable_dropout"] = self.disable_dropout

        config_kwargs = self._filter_kwargs_for_class_or_callable(DPOConfig, config_kwargs)
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
