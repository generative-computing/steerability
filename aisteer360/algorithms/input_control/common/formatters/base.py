"""Token-level and message-level rendering of Memory content."""
from __future__ import annotations

from abc import ABC

import torch
from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.input_control.common.memory.base import Memory


class BaseFormatter(ABC):
    """Renders memory content into an adapted prompt.

    Two entry points: token-level (`apply_to_ids`) and message-level (`apply_to_messages`). Subclasses
    implement one or both; the default of the unused one raises NotImplementedError. Methods using
    `adapt_messages` should prefer `apply_to_messages` since it operates before chat-template tokenization.
    """

    def apply_to_ids(
        self,
        input_ids: torch.Tensor,
        memory: Memory,
        tokenizer: PreTrainedTokenizerBase,
        runtime_kwargs: dict | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement `apply_to_ids`. Use `apply_to_messages` instead."
        )

    def apply_to_messages(
        self,
        messages: list[list[dict]],
        memory: Memory,
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement `apply_to_messages`. Use `apply_to_ids` instead."
        )
