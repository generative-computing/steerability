"""Replace `{{slot}}` placeholders in messages with memory content."""
from __future__ import annotations

import re

from steerability.algorithms.input_control.common.formatters.base import BaseFormatter
from steerability.algorithms.input_control.common.memory.base import Memory


class ChatTemplateSlotFormatter(BaseFormatter):
    """Replaces `{{slot}}` placeholders inside message content with the corresponding memory slot.

    Slot values may be `str` or `list[str]` (lists are joined as bullet points). Unmatched placeholders
    are left as-is.
    """

    _PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

    def __init__(self, list_separator: str = "\n- ", list_prefix: str = "- ") -> None:
        self.list_separator = list_separator
        self.list_prefix = list_prefix

    def _render_value(self, value: str | list[str]) -> str:
        if isinstance(value, list):
            return self.list_prefix + self.list_separator.join(str(v) for v in value)
        return str(value)

    def _substitute(self, content: str, memory: Memory) -> str:
        def _repl(m: re.Match) -> str:
            key = m.group(1)
            if key in memory:
                return self._render_value(memory[key])
            return m.group(0)
        return self._PATTERN.sub(_repl, content)

    def apply_to_messages(
        self,
        messages: list[list[dict]],
        memory: Memory,
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]]:
        out: list[list[dict]] = []
        for chat in messages:
            new_chat: list[dict] = []
            for msg in chat:
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg = {**msg, "content": self._substitute(content, memory)}
                new_chat.append(msg)
            out.append(new_chat)
        return out
