from __future__ import annotations

import gc
import warnings

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.common.processors.value_guided import ValueGuidedProcessor
from aisteer360.algorithms.output_control.common.resolve import resolve_value
from aisteer360.algorithms.output_control.value_guidance.args import ValueGuidanceArgs


class ValueGuidance(OutputControl):
    """Value-guided decoding as configuration: score candidate tokens with a value function and shift their logits.

    `ValueGuidance` is the generic over the step shape. It exposes the `common` value slot through
    flat `Args`: a candidate policy selects a small set of next tokens, a per-candidate value scores
    them, the values are normalized per row, and the selected candidates' logits are shifted by
    `beta * value` (optionally masking non-candidates to `-inf`). A method from the literature is an
    assignment of a config, not a subclass:

        - FUDGE: `value={"kind": "classifier", ...}, policy="top_k", beta=1.0, normalize="none"`.
        - ARGS: `value={"kind": "reward_model", ...}, policy="top_k", normalize="none"`.
        - RAD-equivalent: `value={"kind": "reward_model", ...}, policy="top_k", k=20,
          normalize="clamp", invert=True, mask_non_candidates=True`.
        - SASA-equivalent: `value={"kind": "subspace_margin", ...}, policy="surviving",
          normalize="softmax", mask_non_candidates=False, include_in_scoring=False`.

    `ValueGuidance` is a step-level control: it composes a `ValueGuidedProcessor` into the decoding
    stack, so it works alongside other output controls and any decoding driver. `steer()` resolves
    the value spec (loading a reward model or classifier, or fitting a probe); a fresh processor is
    returned per call.

    Args:
        value: A candidate value: a `BaseCandidateValue` instance, a `(StepContext) -> Tensor[B, K]`
            callable, or a dict spec with a `"kind"` key.
        policy (str): Candidate policy (`"top_k"`, `"top_p"`, `"surviving"`). Defaults to `"top_k"`.
        k (int | None): Candidate count for `policy="top_k"`. Defaults to 20.
        p (float | None): Nucleus threshold for `policy="top_p"`. Defaults to None.
        beta (float): Shift scale. Defaults to 1.0.
        normalize (str): Per-row value normalization (`"none"`, `"minmax"`, `"softmax"`, `"clamp"`).
            Defaults to `"none"`.
        invert (bool): Post-normalization `v <- 1 - v`. Defaults to False.
        mask_non_candidates (bool): Mask non-candidate logits to `-inf`. Defaults to True.
        max_candidates (int | None): Clamp on the candidate set (top-N by score). Defaults to None.
        include_in_scoring (bool): Whether this control's processor also applies during `compute_logprobs`.
            Defaults to True.

    Reference:

    - "FUDGE: Controlled Text Generation With Future Discriminators"
      Kevin Yang, Dan Klein
      [https://arxiv.org/abs/2104.05218](https://arxiv.org/abs/2104.05218)

    - "ARGS: Alignment as Reward-Guided Search"
      Maxim Khanov, Jirayu Burapacheep, Yixuan Li
      [https://arxiv.org/abs/2402.01694](https://arxiv.org/abs/2402.01694)
    """

    Args = ValueGuidanceArgs

    # placeholders (filled by steer)
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    _value = None

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.MODULE`; the value spec resolves against the live model (probe fits,
        placement), which is retained past steer (the generate phase is in-process)."""
        return ModelAccess.MODULE

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | None = None,
        **__,
    ) -> PreTrainedModel:
        """Resolve the value spec, then derive batching / scoring posture from the resolved value."""
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        device = next(model.parameters()).device
        self._value = resolve_value(self.value, model=model, tokenizer=self.tokenizer, device=device)

        self.supports_batching = bool(self._value.supports_batching) and self.policy == "top_k"
        self.same_model_forwards = getattr(self._value, "same_model_forwards", False)
        if self._value.scoring_cost == "model_forward" and self.include_in_scoring:
            warnings.warn(
                "ValueGuidance: this value forwards the pipeline model per reference position during "
                "compute_logprobs; set include_in_scoring=False (SASA's default posture) unless scored "
                "logprobs under the shifted distribution are required.",
                UserWarning,
            )
        return model

    def get_logits_processors(self, input_ids, runtime_kwargs, attention_mask=None, **kwargs) -> list:
        """Return a fresh `ValueGuidedProcessor` for this generation (the freshness contract)."""
        if getattr(self, "_value", None) is None:
            raise RuntimeError("ValueGuidance.steer() must run before generation.")
        return [
            ValueGuidedProcessor(
                self._value,
                policy=self.policy,
                k=self.k,
                p=self.p,
                beta=self.beta,
                normalize=self.normalize,
                invert=self.invert,
                mask_non_candidates=self.mask_non_candidates,
                max_candidates=self.max_candidates,
                lm_tokenizer=self.tokenizer,
                model=self.model,
                attention_mask=attention_mask,
            )
        ]

    def cleanup(self) -> None:
        """Release the resolved value and model references."""
        value = getattr(self, "_value", None)
        if value is not None:
            value.cleanup()
        self._value = None
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
