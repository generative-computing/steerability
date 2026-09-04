from __future__ import annotations

import gc
import logging
import warnings

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.common.loading import load_sequence_classifier
from aisteer360.algorithms.output_control.common.processors.value_guided import ValueGuidedProcessor
from aisteer360.algorithms.output_control.common.values.reward_model import CachedRewardModelValue, RewardModelValue
from aisteer360.algorithms.output_control.rad.args import RADArgs

logger = logging.getLogger(__name__)


class RAD(OutputControl):
    """Implementation of RAD (Reward-Augmented Decoding) from Deng and Raffel, 2023.

    RAD works in two phases:

    1. **Preparation (`steer`)**: load an `AutoModelForSequenceClassification` reward model.
    2. **Controlled decoding (`get_logits_processors`)**: at each decode step, the top-`top_k`
       candidate tokens are scored by the reward model, their reward is clamped to `[0, 1]`, and the
       candidate logits are shifted by `beta * reward` while non-candidate logits are masked to
       `-inf`.

    The score read from the reward model is column `score_index` of its output, after `score_transform`
    (`"none"` reads a raw logit, `"sigmoid"` and `"softmax"` map it into `[0, 1]`). With `invert=True`
    the shift uses `1 - reward`, steering away from the scored attribute. The `"clamp"` normalization
    is absolute rather than relative to the candidate set, so the shift spread across candidates is
    `beta * (max_reward - min_reward)` and benign steps preserve the base distribution. When
    `score_transform="none"` the reward is an unbounded logit, so `invert=True` gives `1 - clamp(v)`,
    which saturates to 0 for any logit at or above 1; pass `score_transform="sigmoid"` to invert a
    logit head meaningfully.

    RAD is a step-level control. `steer()` loads the reward model into a candidate value, and
    `get_logits_processors()` returns a fresh `ValueGuidedProcessor` per call. Candidates are the
    top-`top_k` of the scores this processor receives, so its position in a composed output stack
    matters. Caller-supplied sampling kwargs (temperature, `top_p`, repetition penalty) apply around
    the shift; in particular temperature rescales the effective `beta`, so a protocol-faithful run
    passes `do_sample=True` and nothing else.

    Two scoring paths back the value. When `efficient=True` and the reward model is decoder-only and
    shares the language model's vocabulary, `steer()` builds a `CachedRewardModelValue` that memoizes
    the reward model's prefix activations across steps (the paper's O(km) unidirectional path). When
    a precondition or a steer-time smoke forward fails, RAD emits one `UserWarning` naming the failed
    precondition and falls back to `RewardModelValue(shared_vocab=True)`, which produces the same
    scores at higher cost. When the vocabularies differ, RAD uses `RewardModelValue(shared_vocab=False)`,
    which decodes candidates to text and re-encodes with the reward-model tokenizer. Toggling
    `efficient` changes speed only, not scores.

    Args:
        reward_model_id (str): HF model id or local path for an `AutoModelForSequenceClassification`
            reward model.
        beta (float): Steering intensity (Algorithm 1's beta). Non-negative; direction is set by
            `invert`.
        top_k (int): Number of candidate tokens scored per step (Algorithm 1's k). Defaults to 20.
        invert (bool): Use `1 - reward` as the shift. Defaults to False.
        score_index (int): Output column of the reward model read as the score. Defaults to 0.
        score_transform (str): Map head outputs before selecting `score_index` (`"none"`, `"sigmoid"`,
            or `"softmax"`). Defaults to `"none"`.
        reward_model_kwargs (dict): Extra kwargs for `AutoModelForSequenceClassification.from_pretrained()`.
            Defaults to `{}`.
        include_in_scoring (bool): Apply the processor during `compute_logprobs`. Defaults to True.
        efficient (bool): Cache reward-model prefix activations across steps when preconditions hold.
            Defaults to True.

    Reference:

        - "Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model"
          Haikang Deng, Colin Raffel
          [https://arxiv.org/abs/2310.09520](https://arxiv.org/abs/2310.09520)
    """

    Args = RADArgs

    tokenizer: PreTrainedTokenizer | None = None
    _value = None

    beta: float

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.MODULE`; the reward model's placement follows the live model at steer time
        (the generate phase is in-process)."""
        return ModelAccess.MODULE

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | None = None,
        **__,
    ) -> None:
        """Load the reward model and build the candidate value.

        Loads an `AutoModelForSequenceClassification` reward model and builds a
        `CachedRewardModelValue` when `efficient` is set and the reward model is decoder-only and
        shares the language model's vocabulary (verified by a smoke forward), otherwise a
        `RewardModelValue`. Derives `supports_batching` from the resolved value. Performs no network
        downloads or filesystem writes (model loading may hit the HF cache).

        Args:
            model (PreTrainedModel): The base language model to be steered.
            tokenizer (PreTrainedTokenizer | None): Tokenizer for the base model.
            **__: Additional arguments (unused).
        """
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        device = next(model.parameters()).device

        reward_model, rm_tokenizer = load_sequence_classifier(
            self.reward_model_id,
            device=device,
            hf_model_kwargs=self.reward_model_kwargs,
        )

        shared_vocab = self._vocab_matches(rm_tokenizer, self.tokenizer)
        self._value = self._build_value(reward_model, rm_tokenizer, shared_vocab)
        self.supports_batching = self._value.supports_batching

    def _build_value(self, reward_model, rm_tokenizer, shared_vocab: bool):
        """Select the candidate value: cached (with smoke-test + degrade), shared-vocab, or text."""
        if not shared_vocab:
            return RewardModelValue(
                reward_model, rm_tokenizer,
                score_index=self.score_index, score_transform=self.score_transform,
                shared_vocab=False,
            )

        stateless = RewardModelValue(
            reward_model, rm_tokenizer,
            score_index=self.score_index, score_transform=self.score_transform,
            shared_vocab=True,
        )
        if not self.efficient:
            return stateless

        if not self._is_unidirectional(reward_model):
            warnings.warn(
                "RAD: the reward model is not decoder-only (no past_key_values/cache_position support); "
                "falling back to the stateless reward value.",
                UserWarning,
            )
            return stateless

        cached = CachedRewardModelValue(
            reward_model, rm_tokenizer,
            score_index=self.score_index, score_transform=self.score_transform,
        )
        if not self._cached_smoke_ok(cached):
            warnings.warn(
                "RAD: the cached reward-model forward failed its smoke test; falling back to the "
                "stateless reward value.",
                UserWarning,
            )
            return stateless
        return cached

    @staticmethod
    def _vocab_matches(rm_tokenizer, lm_tokenizer) -> bool:
        """Whether the reward-model and language-model tokenizers share a vocabulary."""
        if lm_tokenizer is None:
            return False
        try:
            return rm_tokenizer.get_vocab() == lm_tokenizer.get_vocab()
        except Exception:
            return False

    @staticmethod
    def _is_unidirectional(reward_model) -> bool:
        """Whether the reward model's forward accepts `past_key_values` (and thus `cache_position`).

        A decoder-only sequence classifier threads `past_key_values` and absorbs `cache_position`
        through a `**kwargs` catch-all; an encoder classifier (BERT/RoBERTa) accepts neither. The
        cached forward's smoke test at steer time is the final gate.
        """
        import inspect

        try:
            params = inspect.signature(reward_model.forward).parameters
        except (TypeError, ValueError):
            return False
        if "past_key_values" not in params:
            return False
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        return "cache_position" in params or has_var_keyword

    def _cached_smoke_ok(self, cached: CachedRewardModelValue) -> bool:
        """Run one tiny cached forward to confirm the reward model supports the cached path."""
        from aisteer360.algorithms.output_control.common.values.base import StepContext

        device = cached._device
        prefix = torch.zeros(1, 2, dtype=torch.long, device=device)
        candidates = torch.zeros(1, 1, dtype=torch.long, device=device)
        ctx = StepContext(
            prefix_ids=prefix,
            candidate_ids=candidates,
            lm_tokenizer=self.tokenizer,
            attention_mask=torch.ones(1, 2, dtype=torch.long, device=device),
        )
        try:
            cached.score(ctx)
        except Exception as exc:
            logger.debug("RAD cached smoke forward failed: %s", exc)
            cached._cached_ids = None
            cached._cache = None
            return False
        cached._cached_ids = None
        cached._cache = None
        return True

    def get_logits_processors(self, input_ids, runtime_kwargs, attention_mask=None, **kwargs) -> list:
        """Return a fresh `ValueGuidedProcessor` implementing RAD's reward-augmented shift.

        Candidates are the top-`top_k` of the scores this processor receives; non-candidate tokens
        are masked to `-inf`; candidate rewards are clamped to `[0, 1]` (inverted when `invert` is
        set) and the candidate logits are shifted by `beta * reward`.
        """
        if self._value is None:
            raise RuntimeError("RAD.steer() must run before generation (reward model not loaded).")
        return [
            ValueGuidedProcessor(
                self._value,
                policy="top_k",
                k=self.top_k,
                beta=self.beta,
                normalize="clamp",
                invert=self.invert,
                mask_non_candidates=True,
                lm_tokenizer=self.tokenizer,
                attention_mask=attention_mask,
            )
        ]

    def cleanup(self) -> None:
        """Release the reward model and tokenizer to free memory."""
        if self._value is not None:
            self._value.cleanup()
        self._value = None
        self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("RAD cleanup completed")
