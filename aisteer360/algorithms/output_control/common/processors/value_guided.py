"""Shift candidate-token logits by a per-candidate value (select candidates, score, combine).

One processor covers many methods. RAD = `(RewardModelValue, top_k, clamp, mask=True)`. SASA =
`(SubspaceMarginValue, surviving, softmax, mask=False)`. FUDGE = `(ClassifierValue, top_k, none,
beta=1)`. ARGS = `(RewardModelValue, top_k, none)`.
"""
from __future__ import annotations

import warnings
from typing import Literal

import torch

from aisteer360.algorithms.output_control.common.candidates import select_candidates
from aisteer360.algorithms.output_control.common.processors.base import PrefixKeyedProcessor
from aisteer360.algorithms.output_control.common.values.base import BaseCandidateValue, StepContext

Normalize = Literal["none", "minmax", "softmax", "clamp"]

LARGE_CANDIDATE_SET_WARN_THRESHOLD = 1024


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
    """`scores[cand] += beta * normalize(value(prefix, cand))`; optionally `-inf` elsewhere.

    Args:
        value: A `BaseCandidateValue`.
        policy: Candidate policy (`"top_k"`, `"top_p"`, `"surviving"`; see `select_candidates`).
        k: Candidate count for `top_k`.
        p: Nucleus threshold for `top_p`.
        beta: Shift scale.
        normalize: `"minmax"` (in-set; degenerate set -> 0.5), `"softmax"` (over the set), `"clamp"`
            (element-wise to `[0, 1]`, absolute rather than in-set), or `"none"`. Applied per row.
        invert: Post-normalization `v <- 1 - v` (steer away from the scored attribute).
        mask_non_candidates: Set non-candidate logits to `-inf` (RAD semantics). Forced False when
            `policy="surviving"` (everything finite is already a candidate).
        max_candidates: Optional clamp on the candidate-set size. After the policy selects candidates,
            keep only the top `max_candidates` by current score. Applies to every policy (it is a
            clamp, not a policy); the main use is bounding the unbounded `surviving` set under a
            model-forward value. `None` (default) disables the clamp.
        lm_tokenizer: The language-model tokenizer, forwarded into `StepContext`.
        model: The pipeline's model (for same-model values), forwarded into `StepContext`.
        attention_mask: The prefix attention mask, forwarded into `StepContext`.
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
        lm_tokenizer=None,
        model=None,
        attention_mask=None,
    ):
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
        self.attention_mask = attention_mask
        self._warned_large_set = False

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
                "value; pass top_k/top_p in gen_kwargs or set max_candidates to bound the per-step cost.",
                UserWarning,
            )
            self._warned_large_set = True

        ctx = StepContext(
            prefix_ids=input_ids,
            candidate_ids=cand_ids,
            lm_tokenizer=self.lm_tokenizer,
            model=self.model,
            attention_mask=self.attention_mask,
        )
        v = self.value.score(ctx).to(scores.dtype)  # [B, K]
        v = _normalize(v, self.normalize, self.invert)

        if self.mask_non_candidates:
            out = torch.full_like(scores, float("-inf"))
            out.scatter_(1, cand_ids, scores.gather(1, cand_ids))
        else:
            out = scores.clone()
        out.scatter_add_(1, cand_ids, self.beta * v)
        return out
