"""Set or merge the leading system message from `memory["instruction"]`."""
from __future__ import annotations

import warnings

import torch
from transformers import PreTrainedTokenizerBase

from steerability.algorithms.input_control.common.formatters.base import BaseFormatter
from steerability.algorithms.input_control.common.memory.base import Memory

_MODES = frozenset({"replace", "prepend", "append"})


class SystemPromptFormatter(BaseFormatter):
    """Sets or merges the leading system message in each chat with `memory["instruction"]`.

    Only the leading system message (`chat[0]` when its role is `"system"`) participates; system messages
    elsewhere in the chat are left untouched. When a leading system message is present, `mode` controls how
    the instruction combines with its content:

        - `"replace"`: the instruction becomes the content.
        - `"prepend"`: `instruction + separator + existing`.
        - `"append"`: `existing + separator + instruction`.

    When no leading system message is present, all three modes insert a single
    `{"role": "system", "content": instruction}` at position 0, so `separator` has no effect. Message dicts are
    copied, so the caller's structures are not mutated. Every input shape yields exactly one leading system
    message.

    The token path (`apply_to_ids`) decodes, re-templates, and re-encodes, so it cannot merge with a system
    prompt that decoding has flattened into text; `mode` does not apply there, and every mode behaves as
    `"replace"`. Prefer message-level entry for chat input.

    Args:
        mode: How the instruction combines with an existing leading system message (`"replace"` (default),
            `"prepend"`, or `"append"`). Ignored when no leading system message is present.
        separator: String inserted between the instruction and the existing content for `"prepend"` and
            `"append"`. Empty string allowed.

    Raises:
        ValueError: If `mode` is not one of the supported values.
    """

    def __init__(self, mode: str = "replace", separator: str = "\n\n") -> None:
        if mode not in _MODES:
            raise ValueError(f"SystemPromptFormatter mode must be one of {sorted(_MODES)}; got {mode!r}.")
        self.mode = mode
        self.separator = separator

    def _resolve_instruction(self, memory: Memory) -> str:
        instruction = memory["instruction"] if "instruction" in memory else memory.get("instruction")
        if not isinstance(instruction, str):
            raise TypeError(
                f"SystemPromptFormatter expects memory['instruction'] to be a str; got {type(instruction).__name__}."
            )
        return instruction

    def apply_to_messages(
        self,
        messages: list[list[dict]],
        memory: Memory,
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]]:
        instruction = self._resolve_instruction(memory)

        out: list[list[dict]] = []
        for chat in messages:
            chat = list(chat)
            if chat and chat[0].get("role") == "system":
                existing = chat[0].get("content", "")
                if self.mode == "prepend":
                    content = instruction + self.separator + existing
                elif self.mode == "append":
                    content = existing + self.separator + instruction
                else:
                    content = instruction
                chat[0] = {"role": "system", "content": content}
            else:
                chat.insert(0, {"role": "system", "content": instruction})
            out.append(chat)
        return out

    def apply_to_ids(
        self,
        input_ids: torch.Tensor,
        memory: Memory,
        tokenizer: PreTrainedTokenizerBase,
        runtime_kwargs: dict | None = None,
    ) -> torch.Tensor:
        warnings.warn(
            "SystemPromptFormatter.apply_to_ids decodes → edits → re-tokenizes; prefer message-level entry "
            "(pass chat input to the pipeline so `apply_to_messages` runs).",
            UserWarning,
        )
        instruction = self._resolve_instruction(memory)

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        rebuilt: list[list[int]] = []
        for user_text in decoded:
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                templated = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                rebuilt.append(tokenizer.encode(templated, add_special_tokens=False))
            else:
                rebuilt.append(tokenizer.encode(instruction + "\n\n" + user_text, add_special_tokens=False))

        max_len = max(len(seq) for seq in rebuilt)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded = [seq + [pad_id] * (max_len - len(seq)) for seq in rebuilt]
        return torch.tensor(padded, dtype=input_ids.dtype, device=input_ids.device)
