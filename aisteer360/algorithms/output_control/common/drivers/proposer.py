"""Rollout proposal for `SearchDriver`.

The composed logits and stopping stacks are forwarded into every rollout (each only when non-empty,
following `HFGenerateDriver`'s pattern), so a step-level control such as RAD steers every proposed
continuation.
"""
from __future__ import annotations

import torch
from transformers import PreTrainedModel

from aisteer360.algorithms.output_control.base import stack_generate_kwargs


class SegmentProposer:
    """Propose `n` continuations of up to `segment_len` tokens from a frontier of sequences.

    Args:
        mode: `"beam"` (`num_beams=n, num_return_sequences=n`, DeAL's rollout) or `"sample"`
            (`do_sample=True, num_return_sequences=n`).
    """

    def __init__(self, mode: str = "beam"):
        if mode not in ("beam", "sample"):
            raise ValueError(f"mode must be 'beam' or 'sample', got {mode!r}.")
        self.mode = mode

    def propose(
        self,
        frontier_ids: torch.Tensor,
        *,
        n: int,
        segment_len: int,
        processors,
        criteria,
        model: PreTrainedModel,
        base_generate=None,
        attention_mask: torch.Tensor | None = None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Roll out `n` continuations of `segment_len` tokens from `frontier_ids`.

        Args:
            frontier_ids: Current context `[F, T]` to continue from.
            n: Number of continuations to produce.
            segment_len: Max new tokens per rollout.
            processors: The composed `LogitsProcessorList` (applied to every rollout when non-empty).
            criteria: The composed `StoppingCriteriaList` (applied to every rollout when non-empty).
            model: The language model.
            base_generate: Optional override for the generate callable (else `model.generate`).
            attention_mask: Optional attention mask matching `frontier_ids`.
            **gen_kwargs: Extra generation kwargs (must not contain processor/criteria objects).

        Returns:
            Full sequences `[F * n, T + segment_len]` (or shorter if a rollout stopped early).
        """
        generate = base_generate if base_generate is not None else model.generate

        extra = stack_generate_kwargs(processors, criteria)
        if attention_mask is not None:
            extra["attention_mask"] = attention_mask

        kwargs = dict(gen_kwargs)
        kwargs.update(
            {
                "max_new_tokens": segment_len,
                "num_return_sequences": n,
            }
        )
        if self.mode == "beam":
            kwargs["num_beams"] = n
        else:
            kwargs["do_sample"] = True

        return generate(input_ids=frontier_ids, **extra, **kwargs)
