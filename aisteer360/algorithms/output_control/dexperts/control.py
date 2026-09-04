from __future__ import annotations

import gc
import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.common.logit_sources import AuxModelSource
from aisteer360.algorithms.output_control.common.processors.contrastive_mixture import ContrastiveMixtureProcessor
from aisteer360.algorithms.output_control.dexperts.args import DExpertsArgs

logger = logging.getLogger(__name__)


class DExperts(OutputControl):
    """
    Implementation of DExperts (Decoding-time Experts) from Liu et al., 2021.

    DExperts steers a base language model at decoding time by combining it with a small "expert" and
    "anti-expert" LM, both fine-tuned on the target attribute (e.g. non-toxic vs. toxic text). At each
    step the next-token distribution is re-weighted in log-prob space by the difference between the
    expert and anti-expert, promoting tokens the expert favors and the anti-expert disfavors.

    DExperts is a step-level control: `steer()` loads the expert and anti-expert into
    `AuxModelSource`s, and `get_logits_processors()` returns a `ContrastiveMixtureProcessor` mixing
    `log p_base + alpha * log p_expert - alpha * log p_anti_expert`. As a step-level control, it
    composes with other output controls and with a decoding driver. The auxiliary models must share
    the base vocabulary (enforced by `AuxModelSource`).

    Doc note: proxy-tuning (Liu et al., 2024) is this control with the expert set to a tuned small
    model and the anti-expert to its untuned counterpart, a documented recipe rather than a separate
    class.

    Args:
        expert_name_or_path (str): HF hub id or local path for the expert LM.
        anti_expert_name_or_path (str): HF hub id or local path for the anti-expert LM.
        alpha (float): Contrast strength. Defaults to 1.0.
        hf_model_kwargs (dict): Extra kwargs passed to `AutoModelForCausalLM.from_pretrained()` for
            both auxiliary models. Defaults to {}.

    Reference:

    - "DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts"
      Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A. Smith,
      Yejin Choi
      [https://arxiv.org/abs/2105.03023](https://arxiv.org/abs/2105.03023)
    """

    Args = DExpertsArgs

    # placeholders (filled by steer)
    tokenizer: PreTrainedTokenizer | None = None
    _expert_source: AuxModelSource | None = None
    _anti_expert_source: AuxModelSource | None = None

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.MODULE`; the expert and anti-expert placements follow the live
        model, whose vocabulary the shared-vocab check reads (the generate phase is
        in-process)."""
        return ModelAccess.MODULE

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | None = None,
        **__,
    ) -> PreTrainedModel:
        """Load the expert and anti-expert into `AuxModelSource`s (shared-vocab enforced)."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self._expert_source = AuxModelSource(
            self.expert_name_or_path,
            base_tokenizer=self.tokenizer,
            hf_model_kwargs=self.hf_model_kwargs,
        )
        self._anti_expert_source = AuxModelSource(
            self.anti_expert_name_or_path,
            base_tokenizer=self.tokenizer,
            hf_model_kwargs=self.hf_model_kwargs,
        )
        self._expert_source.prepare(model=model, tokenizer=self.tokenizer)
        self._anti_expert_source.prepare(model=model, tokenizer=self.tokenizer)
        return model

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs) -> list:
        """Return a fresh `ContrastiveMixtureProcessor` implementing DExperts' expert/anti-expert mix."""
        if self._expert_source is None or self._anti_expert_source is None:
            raise RuntimeError("DExperts.steer() must run before generation (auxiliary models not loaded).")
        return [
            ContrastiveMixtureProcessor(
                sources=[
                    (self._expert_source, self.alpha),
                    (self._anti_expert_source, -self.alpha),
                ],
                base_weight=1.0,
                alpha=None,
            )
        ]

    def cleanup(self) -> None:
        """Release the auxiliary models to free memory."""
        for attr in ("_expert_source", "_anti_expert_source"):
            source = getattr(self, attr, None)
            if source is not None:
                source.cleanup()
                source.model = None
            setattr(self, attr, None)
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("DExperts cleanup completed")
