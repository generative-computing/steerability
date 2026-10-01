"""Shift candidate-token logits by a per-candidate value (select candidates, score, combine).

One processor covers many methods. RAD = `(RewardModelValue, top_k, clamp, mask=True)`. SASA =
`(SubspaceMarginValue, surviving, softmax, mask=False)`. FUDGE = `(ClassifierValue, top_k, none,
beta=1)`. ARGS = `(RewardModelValue, top_k, none)`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.output_control.common.candidates import select_candidates
from steerability.algorithms.output_control.common.kv_cache import full_prefix_mask
from steerability.algorithms.output_control.common.processors.base import PrefixKeyedProcessor
from steerability.algorithms.output_control.common.values.base import BaseCandidateValue, StepContext

Normalize = Literal["none", "minmax", "softmax", "clamp"]

LARGE_CANDIDATE_SET_WARN_THRESHOLD = 1024


@dataclass(frozen=True, slots=True)
class ValueStepRecord:
    """One `ValueGuidedProcessor.process` step, recorded for the caller-owned trace.

    All tensors are detached and on CPU. `candidate_scores` are the processor's input scores for
    the candidates before the shift; `normalized` is the value after `normalize` and `invert`, i.e.
    the quantity the processor multiplies by `beta`.

    Attributes:
        prefix_length: The prefix length at this step (`input_ids.size(1)`).
        candidate_ids: Selected candidate token ids `[B, K]`.
        candidate_scores: Input scores of the candidates before the shift `[B, K]`.
        values: Raw per-candidate value output `[B, K]`.
        normalized: Values after `normalize` and `invert` `[B, K]`.
    """

    prefix_length: int
    candidate_ids: torch.Tensor
    candidate_scores: torch.Tensor
    values: torch.Tensor
    normalized: torch.Tensor


def _normalize(v: torch.Tensor, mode: Normalize, invert: bool) -> torch.Tensor:
    """Normalize per-candidate values row-wise, then optionally invert.

    Args:
        v: Values `[B, K]`.
        mode: `"minmax"` (per-row min-max, relative to the set; degenerate row -> 0.5), `"softmax"`
            (per-row softmax), `"clamp"` (element-wise clamp to `[0, 1]`, absolute rather than
            relative to the set), or `"none"`.
        invert: When True, `v <- 1 - v` after normalization (steer away from the scored attribute).

    Returns:
        Normalized values `[B, K]`.
    """
    if mode == "clamp":
        normalized = v.clamp(0.0, 1.0)
    elif mode == "minmax":
        r_min = v.min(dim=-1, keepdim=True).values
        r_max = v.max(dim=-1, keepdim=True).values
        span = r_max - r_min
        normalized = torch.where(span > 1e-8, (v - r_min) / span.clamp_min(1e-8), 0.5)
    elif mode == "softmax":
        normalized = torch.softmax(v, dim=-1)
    elif mode == "none":
        normalized = v
    else:
        raise ValueError(f"Unknown normalize mode: {mode!r}.")

    if invert:
        normalized = 1 - normalized
    return normalized


class ValueGuidedProcessor(PrefixKeyedProcessor):
    """Logits processor that shifts the scores of candidate tokens by a normalized value.

    At each step, the processor selects candidate tokens from `scores` with `policy`, and `value` scores each
    candidate given the prefix. The processor normalizes the values in each row (`normalize`) and optionally inverts
    them (`invert`). It then adds `beta` times the result to the scores of the candidates. With
    `mask_non_candidates=True`, the scores of all other tokens are set to `-inf`.

    `value` receives an attention mask for the prefix in the `StepContext`. A prefix row that begins with a row of
    `prompt_ids` receives the mask of that prompt row from `attention_mask`, extended with ones over the generated
    positions. Without `prompt_ids`, `attention_mask` is extended in the same way and applies row by row when it has
    as many rows as the prefix. Every other row receives a mask that excludes its leading pad tokens and includes
    every later position, which matches the mask of a left-padded prompt extended over its generated tokens.

    Args:
        value: The `BaseCandidateValue` that scores each candidate given the prefix.
        policy: How candidates are selected, one of `"top_k"` (the `k` highest scores in each row), `"top_p"` (the
            smallest set of tokens whose cumulative probability is at least `p`), or `"surviving"` (every token with
            a finite score). `"top_p"` and `"surviving"` support a batch of one row only.
        k: The number of candidates for `"top_k"`.
        p: The cumulative probability threshold for `"top_p"`.
        beta: The scale applied to the normalized values before they are added to the scores.
        normalize: How the values are normalized in each row, one of `"minmax"` (rescaled to `[0, 1]` over the
            candidate set, with 0.5 for every candidate when the values are equal), `"softmax"` (a softmax over the
            candidate set), `"clamp"` (each value clamped to `[0, 1]` independently of the set), or `"none"`.
        invert: Whether to replace each normalized value `v` with `1 - v`, which steers away from the scored
            attribute.
        mask_non_candidates: Whether to set the scores of tokens outside the candidate set to `-inf`. It is set to
            False when `policy="surviving"`, since every token with a finite score is then a candidate.
        max_candidates: The maximum number of candidates. When the policy selects more, only the `max_candidates`
            candidates with the highest scores are kept. The limit applies to every policy, and it mainly bounds the
            `"surviving"` set when `value` runs a forward pass of the model. None (the default) sets no limit.
        lm_tokenizer: The language model's tokenizer, passed to `value` in the `StepContext`. Its `pad_token_id`
            identifies the leading pad tokens when a row's mask is derived from its tokens.
        model: The pipeline's model, passed to `value` in the `StepContext` for values that run the same model.
        prompt_ids: The prompt token ids of shape `[B_p, P]` that `attention_mask` describes, or None.
        attention_mask: The prompt attention mask of shape `[B_p, P]`, or None.
        trace: A list owned by the caller that receives one `ValueStepRecord` per `process` call, or None (the
            default) to record nothing. The processor only appends to the list and never reads it. Calls made by
            `compute_logprobs` also append records when the control that owns the processor has
            `include_in_scoring=True`.

    Raises:
        ValueError: If `prompt_ids` and `attention_mask` are both given and their shapes differ.
    """

    def __init__(
        self,
        value: BaseCandidateValue,
        *,
        policy: str = "top_k",
        k: int | None = 20,
        p: float | None = None,
        beta: float = 1.0,
        normalize: Normalize = "none",
        invert: bool = False,
        mask_non_candidates: bool = True,
        max_candidates: int | None = None,
        lm_tokenizer: PreTrainedTokenizerBase | None = None,
        model: PreTrainedModel | None = None,
        prompt_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        trace: list | None = None,
    ):
        if prompt_ids is not None and attention_mask is not None and prompt_ids.shape != attention_mask.shape:
            raise ValueError(
                f"prompt_ids {tuple(prompt_ids.shape)} and attention_mask {tuple(attention_mask.shape)} "
                "must have the same shape."
            )
        super().__init__()
        self.value = value
        self.policy = policy
        self.k = k
        self.p = p
        self.beta = beta
        self.normalize = normalize
        self.invert = invert
        self.mask_non_candidates = mask_non_candidates and policy != "surviving"
        self.max_candidates = max_candidates
        self.lm_tokenizer = lm_tokenizer
        self.model = model
        self.prompt_ids = prompt_ids
        self.attention_mask = attention_mask
        self.trace = trace
        self._warned_large_set = False

    def _step_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the attention mask for the prefix at this step.

        A row that begins with a row of `prompt_ids` (pad tokens included) receives the mask of the first such prompt
        row from `attention_mask`, extended with ones to the prefix length. Without `prompt_ids`, `attention_mask` is
        extended in the same way and applies row by row when it has as many rows as `input_ids`. The mask of every
        other row excludes its leading pad tokens and includes every later position. The mask of a row that is
        entirely pad, or of any row when the tokenizer has no pad token, includes every position. Every row's mask is
        derived from its tokens when `attention_mask` is None or longer than the prefix.

        Args:
            input_ids: The prefix token ids of shape `[B, T]`.

        Returns:
            An attention mask of shape `[B, T]` and dtype `torch.long`.
        """
        pad_token_id = getattr(self.lm_tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            derived = torch.ones_like(input_ids, dtype=torch.long)
        else:
            started = (input_ids != pad_token_id).cumsum(dim=1) > 0
            started[~started.any(dim=1)] = True
            derived = started.long()

        stored = self.attention_mask
        if stored is None or stored.size(1) > input_ids.size(1):
            return derived
        stored = stored.to(device=input_ids.device, dtype=torch.long)
        if self.prompt_ids is None:
            return full_prefix_mask(input_ids, stored) if stored.size(0) == input_ids.size(0) else derived
        prompt_ids = self.prompt_ids.to(input_ids.device)
        # [B, B_p]: whether each prefix row begins with each prompt row
        matches = (input_ids[:, None, : prompt_ids.size(1)] == prompt_ids[None]).all(dim=-1)
        prompt_masks = full_prefix_mask(input_ids, stored[matches.long().argmax(dim=1)])
        return torch.where(matches.any(dim=1, keepdim=True), prompt_masks, derived)

    def process(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        cand_ids, cand_scores = select_candidates(scores, self.policy, k=self.k, p=self.p)

        if self.max_candidates is not None and cand_ids.size(1) > self.max_candidates:
            keep = torch.topk(cand_scores, self.max_candidates, dim=-1).indices
            cand_ids = cand_ids.gather(1, keep)

        num_candidates = cand_ids.size(1)
        if (
            not self._warned_large_set
            and num_candidates > LARGE_CANDIDATE_SET_WARN_THRESHOLD
            and getattr(self.value, "scoring_cost", None) == "model_forward"
        ):
            warnings.warn(
                f"ValueGuidedProcessor is evaluating {num_candidates} candidates with a model-forward "
                "value; select a top_k or top_p candidate policy on the control, or set its max_candidates, "
                "to bound the per-step cost. The top_k and top_p generation kwargs do not bound it, since "
                "they apply after the logits processors.",
                UserWarning,
            )
            self._warned_large_set = True

        ctx = StepContext(
            prefix_ids=input_ids,
            candidate_ids=cand_ids,
            lm_tokenizer=self.lm_tokenizer,
            model=self.model,
            attention_mask=self._step_attention_mask(input_ids),
        )
        raw = self.value.score(ctx).to(device=scores.device, dtype=scores.dtype)  # [B, K]
        v = _normalize(raw, self.normalize, self.invert)

        if self.trace is not None:
            self.trace.append(
                ValueStepRecord(
                    prefix_length=int(input_ids.size(1)),
                    candidate_ids=cand_ids.detach().cpu(),
                    candidate_scores=scores.gather(1, cand_ids).detach().cpu(),
                    values=raw.detach().cpu(),
                    normalized=v.detach().cpu(),
                )
            )

        if self.mask_non_candidates:
            out = torch.full_like(scores, float("-inf"))
            out.scatter_(1, cand_ids, scores.gather(1, cand_ids))
        else:
            out = scores.clone()
        out.scatter_add_(1, cand_ids, self.beta * v)
        return out
