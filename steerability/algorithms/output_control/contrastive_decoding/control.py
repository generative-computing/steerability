from __future__ import annotations

import gc
import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.common.logit_sources import AuxModelSource
from steerability.algorithms.output_control.common.processors.contrastive_mixture import ContrastiveMixtureProcessor
from steerability.algorithms.output_control.contrastive_decoding.args import ContrastiveDecodingArgs

logger = logging.getLogger(__name__)


class ContrastiveDecoding(OutputControl):
    """
    Implementation of Contrastive Decoding from Li et al., 2022.

    Contrastive Decoding improves open-ended generation quality by contrasting a strong "expert" (the
    base model) against a weaker "amateur" LM: it favors tokens the expert assigns higher probability
    than the amateur, suppressing the degenerate, repetitive patterns both models share. A
    plausibility mask restricts the contrast to tokens the expert already considers likely
    (`p_base(t) >= alpha * max_t p_base(t)`), preventing the contrast from amplifying implausible
    tokens.

    Contrastive Decoding is a step-level control: `steer()` loads the amateur into an
    `AuxModelSource`, and `get_logits_processors()` returns a `ContrastiveMixtureProcessor` computing
    `base_weight * log p_base - amateur_weight * log p_amateur` over the alpha-masked plausible set.
    As a step-level control, it composes with other output controls and with a decoding driver.
    The amateur must share the base vocabulary (enforced by `AuxModelSource`).

    Reference:

    - "Contrastive Decoding: Open-ended Text Generation as Optimization"
      Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto,
      Luke Zettlemoyer, Mike Lewis
      [https://arxiv.org/abs/2210.15097](https://arxiv.org/abs/2210.15097)
    """

    Args = ContrastiveDecodingArgs

    # placeholders (filled by steer)
    tokenizer: PreTrainedTokenizerBase | None = None
    _amateur_source: AuxModelSource | None = None

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.MODULE`; the amateur's placement follows the live model, whose
        vocabulary the shared-vocab check reads (the generate phase is in-process)."""
        return ModelAccess.MODULE

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Load the amateur into an `AuxModelSource` (shared-vocab enforced)."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self._amateur_source = AuxModelSource(
            self.amateur_name_or_path,
            base_tokenizer=self.tokenizer,
            hf_model_kwargs=self.hf_model_kwargs,
        )
        self._amateur_source.prepare(model=model, tokenizer=self.tokenizer)
        return model

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs) -> list:
        """Return a fresh `ContrastiveMixtureProcessor` implementing the expert-minus-amateur contrast."""
        if self._amateur_source is None:
            raise RuntimeError("ContrastiveDecoding.steer() must run before generation (amateur not loaded).")
        return [
            ContrastiveMixtureProcessor(
                sources=[(self._amateur_source, -self.amateur_weight)],
                base_weight=self.base_weight,
                alpha=self.alpha,
            )
        ]

    def cleanup(self) -> None:
        """Release the amateur model to free memory."""
        if self._amateur_source is not None:
            self._amateur_source.cleanup()
            self._amateur_source.model = None
            self._amateur_source = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("ContrastiveDecoding cleanup completed")
