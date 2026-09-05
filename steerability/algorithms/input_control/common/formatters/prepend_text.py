"""Prepend a raw text block to a user turn or input_ids."""
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from steerability.algorithms.input_control.common.formatters.base import BaseFormatter
from steerability.algorithms.input_control.common.memory.base import Memory

_TARGETS = frozenset({"first_user", "last_user", "all_user"})


class PrependTextFormatter(BaseFormatter):
    """Prepends `memory["text"]` to a user message or to the raw token ids.

    When operating on messages, the prepended text is concatenated to the content of the user-role
    message(s) selected by `target`, joined with `separator`. `target` is one of:

        - `"first_user"`: the first user-role message.
        - `"last_user"`: the last user-role message.
        - `"all_user"`: every user-role message.

    When a chat contains no user-role message, all targets append a new `{"role": "user", "content": text}`
    turn. Message dicts are copied, so the caller's structures are not mutated. When operating on token ids,
    the encoded text is prefixed to each sequence in the batch; token streams carry no turn structure, so
    `target` does not apply on the `apply_to_ids` path.

    Args:
        separator: String inserted between the prepended text and the existing message content.
        target: Which user-role message(s) to prepend to (`"first_user"`, `"last_user"`, or `"all_user"`).

    Raises:
        ValueError: If `target` is not one of the supported values.
    """

    def __init__(self, separator: str = "\n\n", target: str = "first_user") -> None:
        if target not in _TARGETS:
            raise ValueError(
                f"PrependTextFormatter target must be one of {sorted(_TARGETS)}; got {target!r}."
            )
        self.separator = separator
        self.target = target

    def _resolve_text(self, memory: Memory) -> str:
        text = memory["text"] if "text" in memory else memory.get("text")
        if not isinstance(text, str):
            raise TypeError(
                f"PrependTextFormatter expects memory['text'] to be a str; got {type(text).__name__}."
            )
        return text

    def _target_indices(self, chat: list[dict]) -> list[int]:
        user_indices = [i for i, m in enumerate(chat) if m.get("role") == "user"]
        if not user_indices:
            return []
        if self.target == "first_user":
            return [user_indices[0]]
        if self.target == "last_user":
            return [user_indices[-1]]
        return user_indices

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
            indices = self._target_indices(chat)
            if not indices:
                chat.append({"role": "user", "content": text})
            else:
                for idx in indices:
                    chat[idx] = {
                        "role": "user",
                        "content": text + self.separator + chat[idx].get("content", ""),
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
