"""Token-level and message-level rendering of Memory content."""
from __future__ import annotations

from abc import ABC

import torch
from transformers import PreTrainedTokenizerBase

from steerability.algorithms.input_control.common.memory.base import Memory


def insert_prefix_ids(
    input_ids: torch.Tensor, prefix_ids: list[int], tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """Insert `prefix_ids` into each row after the row's leading pad tokens and leading BOS tokens.

    A row of a left-padded batch keeps its pad run in front of the inserted ids, and a BOS token that the
    tokenizer adds stays the first real token. Every row grows by `len(prefix_ids)`, which keeps the batch
    rectangular.

    Args:
        input_ids: Token ids of shape `[B, T]`.
        prefix_ids: The ids to insert.
        tokenizer: Tokenizer whose `pad_token_id` and `bos_token_id` identify the leading tokens. Either id may be
            None, in which case those tokens are not skipped.

    Returns:
        Token ids of shape `[B, T + len(prefix_ids)]`.
    """
    prefix = torch.tensor(prefix_ids, dtype=input_ids.dtype, device=input_ids.device)
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    rows = []
    for row in input_ids:
        start = 0
        if pad_token_id is not None:
            real = (row != pad_token_id).nonzero()
            start = int(real[0]) if real.numel() else 0
        while bos_token_id is not None and start < row.numel() and int(row[start]) == bos_token_id:
            start += 1
        rows.append(torch.cat([row[:start], prefix, row[start:]]))
    return torch.stack(rows)


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
