"""Candidate values from an auxiliary sequence classifier.

For each row-candidate, scores `prefix + candidate` with an auxiliary reward model. Loading and
configuration of the reward model is done by the owning control (RAD), which constructs the value
with the loaded model and tokenizer.

Two values live here. `RewardModelValue` scores each step statelessly. When the reward model shares
the language model's vocabulary (`shared_vocab=True`) it feeds the raw `prefix + candidate` ids to
the reward model; otherwise it decodes to text and re-encodes with the reward-model tokenizer (the
mismatched-vocabulary fallback). `CachedRewardModelValue` produces the same scores as
`RewardModelValue(shared_vocab=True)` at lower cost by memoizing the reward model's prefix
`past_key_values` across decode steps.
"""
from __future__ import annotations

from typing import Any, Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.output_control.common.kv_cache import extends_prefix, full_prefix_mask, repeat_cache
from steerability.algorithms.output_control.common.values.base import BaseCandidateValue, StepContext

ScoreTransform = Literal["none", "sigmoid", "softmax"]


def extract_score(output: Any, score_index: int, score_transform: ScoreTransform) -> torch.Tensor:
    """Read a per-row scalar score from a reward-model output.

    Takes `output.logits` when present, else the raw output tensor `[N, C]`. Applies the transform
    over the class dimension, then selects column `score_index`:

        - `"none"`: select the column directly (a raw, possibly unbounded logit).
        - `"sigmoid"`: element-wise sigmoid, then select.
        - `"softmax"`: softmax over the class dimension, then select.

    Args:
        output: The reward model's output (a classifier output with `logits`, or a raw tensor).
        score_index: Column of the class dimension read as the score.
        score_transform: One of `"none"`, `"sigmoid"`, `"softmax"`.

    Returns:
        A tensor `[N]` of per-row scores.
    """
    logits = output.logits if hasattr(output, "logits") else output
    if score_transform == "sigmoid":
        logits = torch.sigmoid(logits)
    elif score_transform == "softmax":
        logits = torch.softmax(logits, dim=-1)
    return logits[:, score_index]


class RewardModelValue(BaseCandidateValue):
    """Score `prefix + candidate` with an auxiliary reward model (RAD).

    When `shared_vocab` is True, the prefix and candidate ids are fed to the reward model directly
    (no text round-trip), so the reward model scores exactly the sequence the language model
    generated. When False, the ids are decoded to text and re-encoded with the reward-model
    tokenizer, which is the correct behavior for a reward model whose vocabulary differs from the
    language model's. The two paths are not numerically equivalent (the text round-trip may drop
    special tokens and re-segment the boundary token), so a reward model that shares the vocabulary
    should use the id path. On the id path a reward model with a `score` head is read at the
    candidate position explicitly (the last position of every row), so a candidate equal to the
    reward model's pad id is scored as a candidate rather than pooled onto the prefix.

    Args:
        reward_model: The loaded auxiliary reward model (in eval mode).
        rm_tokenizer: Tokenizer for the reward model. Its `max_length` attribute (if set) bounds the
            reward-model input length. On the id path (`shared_vocab=True`), over-length prefixes are
            left-truncated (the tail is kept so the candidate token survives), unlike the text path's
            inherited right-truncation.
        score_index: Column of the reward model's output read as the score.
        score_transform: Map the reward model's output to a score before selecting `score_index`
            (`"none"`, `"sigmoid"`, or `"softmax"`).
        shared_vocab: When True, score via the raw-id path (requires the reward-model vocabulary to
            equal the language model's). When False (default), score via the text round-trip.

    Note:
        `scoring_cost="aux_forward"` and `supports_batching=True`; candidate scoring batches `B * K`
        rows through the auxiliary model.
    """

    scoring_cost = "aux_forward"

    def __init__(
        self,
        reward_model: PreTrainedModel,
        rm_tokenizer: PreTrainedTokenizerBase,
        score_index: int = 0,
        score_transform: ScoreTransform = "none",
        shared_vocab: bool = False,
    ):
        self.reward_model = reward_model
        self.rm_tokenizer = rm_tokenizer
        self.score_index = score_index
        self.score_transform = score_transform
        self.shared_vocab = shared_vocab
        self.supports_batching = True
        self._device = next(reward_model.parameters()).device

    @torch.inference_mode()
    def score(self, ctx: StepContext) -> torch.Tensor:
        if self.shared_vocab:
            return self._score_ids(ctx)
        return self._score_text(ctx)

    def _score_text(self, ctx: StepContext) -> torch.Tensor:
        """Score by decoding `prefix + candidate` to text and re-encoding (mismatched-vocab path).

        The ids are decoded with `skip_special_tokens=True`, so a special-token candidate such as
        eos contributes nothing to the scored text.
        """
        batch_size = ctx.prefix_ids.size(0)
        num_candidates = ctx.candidate_ids.size(1)

        prefix = ctx.prefix_ids.unsqueeze(1).expand(-1, num_candidates, -1)  # [B, K, T]
        combined = torch.cat([prefix, ctx.candidate_ids.unsqueeze(-1)], dim=-1)  # [B, K, T+1]
        flat = combined.reshape(batch_size * num_candidates, -1)  # [B*K, T+1]
        texts = ctx.lm_tokenizer.decode(flat, skip_special_tokens=True)

        max_length = getattr(self.rm_tokenizer, "max_length", None)
        inputs = self.rm_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self._device)
        output = self.reward_model(**inputs)
        rewards = extract_score(output, self.score_index, self.score_transform)  # [B*K]
        return rewards.reshape(batch_size, num_candidates)

    def _score_ids(self, ctx: StepContext) -> torch.Tensor:
        """Score by feeding raw `prefix + candidate` ids to the reward model (shared-vocab path)."""
        batch_size = ctx.prefix_ids.size(0)
        num_candidates = ctx.candidate_ids.size(1)

        prefix_mask = full_prefix_mask(ctx.prefix_ids, ctx.attention_mask)  # [B, T]
        prefix = ctx.prefix_ids.unsqueeze(1).expand(-1, num_candidates, -1)  # [B, K, T]
        combined = torch.cat([prefix, ctx.candidate_ids.unsqueeze(-1)], dim=-1)  # [B, K, T+1]
        mask = torch.cat(
            [prefix_mask.unsqueeze(1).expand(-1, num_candidates, -1),
             torch.ones(batch_size, num_candidates, 1, device=prefix_mask.device, dtype=prefix_mask.dtype)],
            dim=-1,
        )  # [B, K, T+1]
        ids = combined.reshape(batch_size * num_candidates, -1).to(self._device)  # [B*K, T+1]
        mask = mask.reshape(batch_size * num_candidates, -1).to(self._device)

        max_length = getattr(self.rm_tokenizer, "max_length", None)
        if max_length is not None and ids.size(1) > max_length:
            ids = ids[:, -max_length:]
            mask = mask[:, -max_length:]

        if hasattr(self.reward_model, "score"):
            # the head is applied at the last position, which is the candidate for every row; the
            # classifier's own forward pools at the last id not equal to the configured pad id
            backbone = getattr(self.reward_model, self.reward_model.base_model_prefix)
            hidden = backbone(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            output = self.reward_model.score(hidden[:, -1, :])
        else:
            output = self.reward_model(input_ids=ids, attention_mask=mask)
        rewards = extract_score(output, self.score_index, self.score_transform)  # [B*K]
        return rewards.reshape(batch_size, num_candidates)


class CachedRewardModelValue(BaseCandidateValue):
    """Score `prefix + candidate` with a unidirectional reward model, caching prefix activations.

    Produces the same scores as `RewardModelValue(shared_vocab=True)` under the same `score_index`
    and `score_transform`, differing only in cost. The reward model's prefix `past_key_values` are
    memoized keyed on the prefix ids: each step forwards only the delta tokens to extend the cache,
    then evaluates the K candidates as single-token forwards over a repeated copy of the cache. Any
    non-extension of the cached prefix (rewind, reorder, restart, teacher-forced replay) rebuilds the
    cache from scratch, which reproduces a fresh full forward.

    The reward model must be decoder-only (accept `past_key_values` and `cache_position`) and share
    the language model's vocabulary; the owning control checks both preconditions and a smoke forward
    at steer time and falls back to `RewardModelValue(shared_vocab=True)` on failure. Batch size 1
    only.

    Args:
        reward_model: The loaded unidirectional reward model (in eval mode).
        rm_tokenizer: Tokenizer for the reward model. Its `max_length` attribute (if set) bounds the
            reward-model input length; over-length prefixes are scored without the cache over the
            left-truncated tail.
        score_index: Column of the reward model's output read as the score.
        score_transform: Map the reward model's output to a score before selecting `score_index`
            (`"none"`, `"sigmoid"`, or `"softmax"`).

    Note:
        `scoring_cost="aux_forward"` and `supports_batching=False`.
    """

    scoring_cost = "aux_forward"

    def __init__(
        self,
        reward_model: PreTrainedModel,
        rm_tokenizer: PreTrainedTokenizerBase,
        score_index: int = 0,
        score_transform: ScoreTransform = "none",
    ):
        self.reward_model = reward_model
        self.rm_tokenizer = rm_tokenizer
        self.score_index = score_index
        self.score_transform = score_transform
        self.supports_batching = False
        self._device = next(reward_model.parameters()).device
        self._cached_ids: torch.Tensor | None = None  # [1, T_c]
        self._cache = None  # past_key_values covering _cached_ids

    def _sync_cache(self, prefix_ids: torch.Tensor, full_mask: torch.Tensor) -> None:
        """Bring the internal cache up to `prefix_ids` (extend by the delta, or rebuild)."""
        if not extends_prefix(self._cached_ids, prefix_ids):
            out = self.reward_model(
                input_ids=prefix_ids, attention_mask=full_mask, use_cache=True, return_dict=True,
                cache_position=torch.arange(prefix_ids.size(1), device=prefix_ids.device),
            )
            self._cache = out.past_key_values
        else:
            cached_len = self._cached_ids.size(1)
            if prefix_ids.size(1) > cached_len:
                delta = prefix_ids[:, cached_len:]
                positions = torch.arange(cached_len, prefix_ids.size(1), device=prefix_ids.device)
                out = self.reward_model(
                    input_ids=delta, attention_mask=full_mask, past_key_values=self._cache,
                    use_cache=True, cache_position=positions, return_dict=True,
                )
                self._cache = out.past_key_values
        self._cached_ids = prefix_ids.detach()

    @torch.inference_mode()
    def score(self, ctx: StepContext) -> torch.Tensor:
        if ctx.prefix_ids.size(0) != 1:
            raise ValueError("CachedRewardModelValue supports batch size 1 only.")

        num_candidates = ctx.candidate_ids.size(1)
        prefix_ids = ctx.prefix_ids.to(self._device)
        candidate_ids = ctx.candidate_ids.to(self._device)
        full_mask = full_prefix_mask(prefix_ids, ctx.attention_mask.to(self._device) if ctx.attention_mask is not None else None)

        max_length = getattr(self.rm_tokenizer, "max_length", None)
        if max_length is not None and prefix_ids.size(1) + 1 > max_length:
            return self._score_no_cache(prefix_ids, candidate_ids, full_mask, max_length)

        self._sync_cache(prefix_ids, full_mask)

        prefix_len = prefix_ids.size(1)
        repeated = repeat_cache(self._cache, num_candidates, preserve_input=True)
        cand_tokens = candidate_ids.reshape(num_candidates, 1)
        cand_mask = torch.cat(
            [full_mask.repeat(num_candidates, 1),
             torch.ones(num_candidates, 1, device=self._device, dtype=full_mask.dtype)],
            dim=1,
        )
        positions = torch.arange(prefix_len, prefix_len + 1, device=self._device)
        output = self.reward_model(
            input_ids=cand_tokens,
            attention_mask=cand_mask,
            past_key_values=repeated,
            use_cache=True,
            cache_position=positions,
            return_dict=True,
        )
        rewards = extract_score(output, self.score_index, self.score_transform)  # [K]
        return rewards.reshape(1, num_candidates)

    def _score_no_cache(
        self,
        prefix_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        full_mask: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """Full forward over the left-truncated `prefix + candidate` for an over-length prefix."""
        num_candidates = candidate_ids.size(1)
        prefix = prefix_ids.expand(num_candidates, -1)
        combined = torch.cat([prefix, candidate_ids.reshape(num_candidates, 1)], dim=-1)  # [K, T+1]
        mask = torch.cat(
            [full_mask.repeat(num_candidates, 1),
             torch.ones(num_candidates, 1, device=self._device, dtype=full_mask.dtype)],
            dim=1,
        )
        combined = combined[:, -max_length:]
        mask = mask[:, -max_length:]
        output = self.reward_model(input_ids=combined, attention_mask=mask, return_dict=True)
        rewards = extract_score(output, self.score_index, self.score_transform)  # [K]
        return rewards.reshape(1, num_candidates)

    def cleanup(self) -> None:
        """Release the reward model and cached activations."""
        self.reward_model = None
        self._cache = None
        self._cached_ids = None
