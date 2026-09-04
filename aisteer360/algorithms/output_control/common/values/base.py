"""`StepContext` and `BaseCandidateValue`, the candidate-value interface.

`values/` scores candidates (per-token scalars for one decode step); the sibling `scorers/` scores
continuations (per-sequence floats). Values are consumed by `ValueGuidedProcessor`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(slots=True)
class StepContext:
    """Everything a candidate value may consult at one decoding step.

    Attributes:
        prefix_ids: The current prefix token ids `[B, T]`.
        candidate_ids: The candidate next-token ids `[B, K]`.
        lm_tokenizer: The language model's tokenizer.
        model: The pipeline's model (only used by same-model values; see `CandidateForward`).
        attention_mask: The prefix attention mask `[B, T]`, when available.
    """

    prefix_ids: torch.Tensor
    candidate_ids: torch.Tensor
    lm_tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel | None = None
    attention_mask: torch.Tensor | None = None


class BaseCandidateValue(ABC):
    """Score candidate continuations of a prefix; higher = more desired.

    `score()` returns a float tensor `[B, K]` aligned with `ctx.candidate_ids`. Values must be pure
    in the `PrefixKeyedProcessor` sense; any caching keys on the prefix.

    Class attributes:
        supports_batching: Whether the value handles `B > 1` candidate rows.
        scoring_cost: Rough cost tier, surfaced so controls can warn when scoring with expensive
            values is enabled. `"cheap"` (no model forward), `"aux_forward"` (an auxiliary-model
            forward), or `"model_forward"` (a forward of the pipeline's own model).
        same_model_forwards: Whether this value issues additional forward passes through the
            pipeline's own model during decoding. Such passes must be wrapped in
            `auxiliary_pass()` (see `aisteer360.algorithms.core.utils.auxiliary_pass`), which
            keeps them out of state-control condition scoring, gate updates, and fallback
            position counting. Defaults to False; the flag is declarative metadata and is not
            read by the pipeline.
    """

    supports_batching: bool = False
    scoring_cost: Literal["cheap", "aux_forward", "model_forward"] = "cheap"
    same_model_forwards: bool = False

    @abstractmethod
    def score(self, ctx: StepContext) -> torch.Tensor:
        """Return per-candidate values `[B, K]` aligned with `ctx.candidate_ids`."""
        ...

    def prepare(self, model=None, tokenizer=None, **kwargs) -> None:
        """Optional offline setup, invoked from the owning control's `steer()`.

        Mirrors the input-control selector `prepare()` convention: a default no-op that the parent
        control calls during its own `steer()` when the value needs preparation.
        """
        pass

    def cleanup(self) -> None:
        """Release any resources allocated during `prepare()`."""
        pass
