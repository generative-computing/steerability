"""Prepend a raw text block to the user turn or input_ids."""
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.input_control.common.formatters.base import BaseFormatter
from aisteer360.algorithms.input_control.common.memory.base import Memory


class PrependTextFormatter(BaseFormatter):
    """Prepends `memory["text"]` either to the first user message or to the raw token ids.

    When operating on messages, the prepended text is concatenated to the content of the first user-role
    message (creating one if none exists). When operating on token ids, the encoded text is prefixed to each
    sequence in the batch.
    """

    def __init__(self, separator: str = "\n\n") -> None:
        self.separator = separator

    def _resolve_text(self, memory: Memory) -> str:
        text = memory["text"] if "text" in memory else memory.get("text")
        if not isinstance(text, str):
            raise TypeError(
                f"PrependTextFormatter expects memory['text'] to be a str; got {type(text).__name__}."
            )
        return text

    def apply_to_messages(
        self,
        messages: list[list[dict]],
        memory: Memory,
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]]:
        text = self._resolve_text(memory)
        out: list[list[dict]] = []
        for chat in messages:
            chat = [dict(m) for m in chat]
            user_idx = next((i for i, m in enumerate(chat) if m.get("role") == "user"), None)
            if user_idx is None:
                chat.append({"role": "user", "content": text})
            else:
                chat[user_idx] = {
                    "role": "user",
                    "content": text + self.separator + chat[user_idx].get("content", ""),
                }
            out.append(chat)
        return out

    def apply_to_ids(
        self,
        input_ids: torch.Tensor,
        memory: Memory,
        tokenizer: PreTrainedTokenizerBase,
        runtime_kwargs: dict | None = None,
    ) -> torch.Tensor:
        text = self._resolve_text(memory)
        prefix_ids = tokenizer.encode(text + self.separator, add_special_tokens=False)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        prefix = torch.tensor(prefix_ids, dtype=input_ids.dtype, device=input_ids.device)
        prefix = prefix.unsqueeze(0).expand(input_ids.size(0), -1)
        return torch.cat([prefix, input_ids], dim=1)
