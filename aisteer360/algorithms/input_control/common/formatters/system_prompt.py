"""Set or replace the system message from `memory["instruction"]`."""
from __future__ import annotations

import warnings

import torch
from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.input_control.common.formatters.base import BaseFormatter
from aisteer360.algorithms.input_control.common.memory.base import Memory


class SystemPromptFormatter(BaseFormatter):
    """Sets or replaces the leading system message in each chat with `memory["instruction"]`."""

    def apply_to_messages(
        self,
        messages: list[list[dict]],
        memory: Memory,
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]]:
        instruction = memory["instruction"] if "instruction" in memory else memory.get("instruction")
        if not isinstance(instruction, str):
            raise TypeError(
                f"SystemPromptFormatter expects memory['instruction'] to be a str; got {type(instruction).__name__}."
            )

        out: list[list[dict]] = []
        for chat in messages:
            chat = list(chat)
            if chat and chat[0].get("role") == "system":
                chat[0] = {"role": "system", "content": instruction}
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
        instruction = memory["instruction"] if "instruction" in memory else memory.get("instruction")
        if not isinstance(instruction, str):
            raise TypeError(
                f"SystemPromptFormatter expects memory['instruction'] to be a str; got {type(instruction).__name__}."
            )

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
