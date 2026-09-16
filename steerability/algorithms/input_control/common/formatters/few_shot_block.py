"""Render `memory['examples']` as a labeled system-message block."""
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from steerability.algorithms.input_control.common.formatters.base import BaseFormatter
from steerability.algorithms.input_control.common.memory.base import Memory

_MODES = frozenset({"append", "prepend", "insert"})


class FewShotBlockFormatter(BaseFormatter):
    """Renders memory['directive'] and memory['examples'] as a single system message.

    Each example dict's non-private fields (those not prefixed with `_`) are emitted as
    `Title-Cased Key: value` lines under a header determined by the example's `_polarity`.
    The `_polarity` key is set internally by `FewShot` and is the only required field.

    Only the leading system message (`chat[0]` when its role is `"system"`) participates in placement;
    messages after it are left untouched. When a leading system message is present, `mode` controls how
    the rendered block combines with it:

        - `"append"`: `existing + separator + block`.
        - `"prepend"`: `block + separator + existing`.
        - `"insert"`: the block becomes a separate second system message at index 1.

    When no leading system message is present, all three modes insert a single
    `{"role": "system", "content": block}` at position 0, so `separator` has no effect. Message dicts are
    copied, so the caller's structures are not mutated. Under `"append"` and `"prepend"` every input shape
    yields exactly one leading system message; `"insert"` yields two, which some chat templates (Qwen3)
    reject.

    The token path (`apply_to_ids`) prepends the block text to the token stream. There is no message
    structure to merge with, so `mode` does not apply there.

    Args:
        positive_header: Header emitted above each positive example.
        negative_header: Header emitted above each negative example.
        mode: How the block combines with an existing leading system message (`"append"` (default),
            `"prepend"`, or `"insert"`). Ignored when no leading system message is present.
        separator: String inserted between the existing content and the block for `"append"` and
            `"prepend"`. Empty string allowed.

    Raises:
        ValueError: If `mode` is not one of the supported values.
        TypeError: If `separator` is not a `str`.
    """

    DEFAULT_POSITIVE_HEADER = "### Positive example (behavior to follow)"
    DEFAULT_NEGATIVE_HEADER = "### Negative example (behavior to avoid)"

    def __init__(
        self,
        positive_header: str = DEFAULT_POSITIVE_HEADER,
        negative_header: str = DEFAULT_NEGATIVE_HEADER,
        mode: str = "append",
        separator: str = "\n\n",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"FewShotBlockFormatter mode must be one of {sorted(_MODES)}; got {mode!r}.")
        if not isinstance(separator, str):
            raise TypeError(f"separator must be a str; got {type(separator).__name__}.")
        self.positive_header = positive_header
        self.negative_header = negative_header
        self.mode = mode
        self.separator = separator

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
            if chat and chat[0].get("role") == "system":
                if self.mode == "insert":
                    chat.insert(1, {"role": "system", "content": block})
                else:
                    existing = chat[0].get("content", "")
                    if self.mode == "prepend":
                        chat[0]["content"] = block + self.separator + existing
                    else:
                        chat[0]["content"] = existing + self.separator + block
            else:
                chat.insert(0, {"role": "system", "content": block})
            out.append(chat)
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
