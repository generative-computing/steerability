"""Wrap a user function as a candidate value.

The escape hatch of the value family: any `(StepContext) -> Tensor[B, K]` callable becomes a
`BaseCandidateValue` without a subclass. The wrapped function is responsible for returning
per-candidate scores aligned with `ctx.candidate_ids`; the batching and cost flags are supplied
explicitly at construction (they cannot be inferred from a bare callable).
"""
from __future__ import annotations

from typing import Callable, Literal

import torch

from aisteer360.algorithms.output_control.common.values.base import BaseCandidateValue, StepContext


class CallableValue(BaseCandidateValue):
    """Adapt a `(StepContext) -> Tensor[B, K]` callable into a `BaseCandidateValue`.

    `same_model_forwards` keeps the `BaseCandidateValue` default of False, since it cannot be
    inferred from a bare callable. A callable that forwards the pipeline's model should be wrapped in
    a component that declares `same_model_forwards`.

    Args:
        fn: A callable mapping a `StepContext` to per-candidate values `[B, K]` aligned with
            `ctx.candidate_ids`; higher = more desired.
        supports_batching: Whether `fn` handles `B > 1` candidate rows. Defaults to False.
        scoring_cost: Cost tier surfaced to owning controls (`"cheap"`, `"aux_forward"`, or
            `"model_forward"`). Defaults to `"cheap"`.
    """

    def __init__(
        self,
        fn: Callable[[StepContext], torch.Tensor],
        *,
        supports_batching: bool = False,
        scoring_cost: Literal["cheap", "aux_forward", "model_forward"] = "cheap",
    ):
        if not callable(fn):
            raise TypeError("CallableValue requires a callable (StepContext) -> Tensor[B, K].")
        self.fn = fn
        self.supports_batching = supports_batching
        self.scoring_cost = scoring_cost

    def score(self, ctx: StepContext) -> torch.Tensor:
        return torch.as_tensor(self.fn(ctx))
