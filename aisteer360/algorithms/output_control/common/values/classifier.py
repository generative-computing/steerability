"""`ClassifierValue` (FUDGE-family) — candidate value from an attribute classifier.

Returns `log p(attribute | prefix + candidate)` per candidate. Accepts either an HF sequence
classifier (with a `label_index` selecting the attribute logit) or any callable mapping a list of
texts to a `[batch]` tensor of attribute log-probabilities.
"""
from __future__ import annotations

from typing import Callable

import torch

from aisteer360.algorithms.output_control.common.values.base import BaseCandidateValue, StepContext


class ClassifierValue(BaseCandidateValue):
    """Score `prefix + candidate` with an attribute classifier (FUDGE-family).

    Args:
        classifier: Either a callable `list[str] -> Tensor[batch]` returning attribute log-probs, or
            an HF sequence classifier (used with `classifier_tokenizer` + `label_index`).
        classifier_tokenizer: Tokenizer for an HF classifier (required when `classifier` is a model).
        label_index: Index of the attribute logit for an HF classifier.

    Note:
        `scoring_cost="aux_forward"`, `supports_batching=True`.
    """

    supports_batching: bool = True
    scoring_cost = "aux_forward"

    def __init__(
        self,
        classifier,
        classifier_tokenizer=None,
        label_index: int = 1,
    ):
        self.classifier = classifier
        self.classifier_tokenizer = classifier_tokenizer
        self.label_index = label_index
        self._callable = not hasattr(classifier, "parameters")
        self._device = None if self._callable else next(classifier.parameters()).device

    @torch.inference_mode()
    def score(self, ctx: StepContext) -> torch.Tensor:
        batch_size = ctx.prefix_ids.size(0)
        num_candidates = ctx.candidate_ids.size(1)

        prefix = ctx.prefix_ids.unsqueeze(1).expand(-1, num_candidates, -1)
        combined = torch.cat([prefix, ctx.candidate_ids.unsqueeze(-1)], dim=-1)
        flat = combined.reshape(batch_size * num_candidates, -1)
        texts = ctx.lm_tokenizer.batch_decode(flat, skip_special_tokens=True)

        if self._callable:
            logp = self.classifier(texts)
            logp = torch.as_tensor(logp, dtype=torch.float32)
        else:
            inputs = self.classifier_tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
            logits = self.classifier(**inputs).logits
            logp = torch.log_softmax(logits, dim=-1)[:, self.label_index]

        return logp.reshape(batch_size, num_candidates)
