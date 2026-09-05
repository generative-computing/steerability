"""Mix full-vocabulary log-prob sources into the base distribution.

`scores' = w0 * log p_base + sum_i w_i * log p_source_i`, with an optional plausibility mask
`keep = {t : p_base(t) >= alpha * max_t p_base(t)}` applied before mixing (contrastive decoding's
alpha-filter; `alpha=None` disables; masked-out tokens -> -inf). Mixing is in log-prob space; the
return is unnormalized scores, which HF's loop accepts.
"""
from __future__ import annotations

import torch

from steerability.algorithms.output_control.common.logit_sources import BaseLogitSource
from steerability.algorithms.output_control.common.processors.base import PrefixKeyedProcessor


class ContrastiveMixtureProcessor(PrefixKeyedProcessor):
    """Mix the base log-probs with weighted auxiliary log-prob sources.

    Args:
        sources: A list of `(BaseLogitSource, weight)` pairs.
        base_weight: Weight `w0` on the base log-probs.
        alpha: Plausibility-mask threshold in `[0, 1]`; keep tokens with
            `p_base(t) >= alpha * max_t p_base(t)`. `None` disables the mask.
    """

    def __init__(
        self,
        sources: list[tuple[BaseLogitSource, float]],
        *,
        base_weight: float = 1.0,
        alpha: float | None = None,
    ):
        super().__init__()
        self.sources = sources
        self.base_weight = base_weight
        self.alpha = alpha

    def reset_state(self, input_ids: torch.Tensor) -> None:
        # sources recompute per call; no per-generation state to reset here,
        # but a source that caches can override its own reset via this hook path.
        pass

    def process(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        base_logprobs = torch.log_softmax(scores, dim=-1)
        mixed = self.base_weight * base_logprobs
        for source, weight in self.sources:
            mixed = mixed + weight * source.logprobs(input_ids).to(mixed.dtype)

        if self.alpha is not None:
            base_probs = base_logprobs.exp()
            threshold = self.alpha * base_probs.max(dim=-1, keepdim=True).values
            keep = base_probs >= threshold
            mixed = mixed.masked_fill(~keep, float("-inf"))
        return mixed
