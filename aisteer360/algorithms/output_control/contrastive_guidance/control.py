from __future__ import annotations

import gc

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.common.processors.contrastive_mixture import ContrastiveMixtureProcessor
from aisteer360.algorithms.output_control.common.resolve import resolve_source
from aisteer360.algorithms.output_control.contrastive_guidance.args import ContrastiveGuidanceArgs


class ContrastiveGuidance(OutputControl):
    """Contrastive mixing of next-token distributions as configuration.

    `ContrastiveGuidance` is the generic over the distribution shape. It exposes the `common` logit
    source slot through flat `Args`: `steer()` resolves a parallel list of sources and mixes their
    log-probs with the base distribution as `base_weight * log p_base + sum_i weights[i] * log
    p_source_i`, with an optional plausibility mask. A method from the literature is an assignment of
    a config:

        - DExperts: `sources=[expert_id, anti_expert_id], weights=[+α, -α]`.
        - Contrastive decoding: `sources=[amateur_id], weights=[-1.0], alpha=0.1`.
        - Proxy-tuning: `sources=[tuned_small_id, untuned_small_id], weights=[+1.0, -1.0]`.
        - CFG / context-aware decoding: `sources=[{"kind": "prompt_variant", ...}], base_weight=gamma,
          weights=[-(gamma-1)]` (a `prompt_variant` source forwards the pipeline's own model; see the
          compatibility note on `PromptVariantSource`).

    `ContrastiveGuidance` is a step-level control: it composes a `ContrastiveMixtureProcessor` into
    the decoding stack, so it works alongside other output controls and any decoding driver.
    Auxiliary sources must share the base vocabulary (enforced by `AuxModelSource`).
    `supports_batching` is False; generation runs one prompt at a time.

    Args:
        sources (list): Source specs (each a `BaseLogitSource` instance, a callable, an aux-model
            name/path string, or a dict spec with a `"kind"` key).
        weights (list[float]): Weights parallel to `sources`.
        base_weight (float): Weight on the base model's log-probs. Defaults to 1.0.
        alpha (float | None): Plausibility-mask threshold in `(0, 1]`; `None` disables. Defaults to None.
        include_in_scoring (bool): Whether this control's processor also applies during `compute_logprobs`.
            Defaults to True.

    Reference:

    - "DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts"
      Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A. Smith,
      Yejin Choi
      [https://arxiv.org/abs/2105.03023](https://arxiv.org/abs/2105.03023)

    - "Contrastive Decoding: Open-ended Text Generation as Optimization"
      Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto,
      Luke Zettlemoyer, Mike Lewis
      [https://arxiv.org/abs/2210.15097](https://arxiv.org/abs/2210.15097)
    """

    Args = ContrastiveGuidanceArgs

    # placeholders (filled by steer)
    tokenizer: PreTrainedTokenizer | None = None
    _sources: list | None = None

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.MODULE`; sources may bind the live model (the prompt-variant source
        forwards it during decoding), which is retained past steer (the generate phase is
        in-process)."""
        return ModelAccess.MODULE

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | None = None,
        **__,
    ) -> PreTrainedModel:
        """Resolve each source spec (loading auxiliary models and enforcing shared vocabularies)."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self._sources = [
            resolve_source(spec, model=model, tokenizer=self.tokenizer) for spec in self.sources
        ]
        self.same_model_forwards = any(
            getattr(source, "same_model_forwards", False) for source in self._sources
        )
        return model

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs) -> list:
        """Return a fresh `ContrastiveMixtureProcessor` mixing the resolved sources."""
        if self._sources is None:
            raise RuntimeError("ContrastiveGuidance.steer() must run before generation (sources not loaded).")
        return [
            ContrastiveMixtureProcessor(
                list(zip(self._sources, self.weights)),
                base_weight=self.base_weight,
                alpha=self.alpha,
            )
        ]

    def cleanup(self) -> None:
        """Release every resolved source to free auxiliary-model memory."""
        for source in self._sources or []:
            source.cleanup()
        self._sources = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
