"""Render `memory['examples']` as a labeled system-message block."""
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.input_control.common.formatters.base import BaseFormatter
from aisteer360.algorithms.input_control.common.memory.base import Memory


class FewShotBlockFormatter(BaseFormatter):
    """Renders memory['directive'] and memory['examples'] as a single system message.

    Each example dict's non-private fields (those not prefixed with `_`) are emitted as
    `Title-Cased Key: value` lines under a header determined by the example's `_polarity`.
    The `_polarity` key is set internally by `FewShot` and is the only required field.
    """

    DEFAULT_POSITIVE_HEADER = "### Positive example (behavior to follow)"
    DEFAULT_NEGATIVE_HEADER = "### Negative example (behavior to avoid)"

    def __init__(
        self,
        positive_header: str = DEFAULT_POSITIVE_HEADER,
        negative_header: str = DEFAULT_NEGATIVE_HEADER,
    ) -> None:
        self.positive_header = positive_header
        self.negative_header = negative_header

    def _render_example(self, example: dict) -> str:
        header = (
            self.negative_header if example.get("_polarity") == "negative"
            else self.positive_header
        )
        body = "\n".join(
            f"{key.replace('_', ' ').title()}: {value}"
            for key, value in example.items()
            if not key.startswith("_")
        )
        return f"{header}\n{body}"

    def _render_block(self, memory: Memory) -> str:
        examples = memory.get("examples") or []
        directive = memory.get("directive") or ""
        parts: list[str] = []
        if directive:
            parts.append(directive)
        parts.extend(self._render_example(ex) for ex in examples)
        return "\n\n".join(parts)

    def apply_to_messages(
        self,
        messages: list[list[dict]],
        memory: Memory,
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]]:
        block = self._render_block(memory)
        if not block:
            return [list(chat) for chat in messages]

        out: list[list[dict]] = []
        for chat in messages:
            chat = [dict(m) for m in chat]
            insert_at = 1 if chat and chat[0].get("role") == "system" else 0
            out.append(chat[:insert_at] + [{"role": "system", "content": block}] + chat[insert_at:])
        return out

    def apply_to_ids(
        self,
        input_ids: torch.Tensor,
        memory: Memory,
        tokenizer: PreTrainedTokenizerBase,
        runtime_kwargs: dict | None = None,
    ) -> torch.Tensor:
        block = self._render_block(memory)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if not block:
            return input_ids
        prefix_ids = tokenizer.encode(block + "\n\n", add_special_tokens=False)
        prefix = torch.tensor(prefix_ids, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([prefix.unsqueeze(0).expand(input_ids.size(0), -1), input_ids], dim=1)
