"""
Input control that sets or merges the leading system message of a chat.
"""
import torch
from transformers import PreTrainedTokenizerBase

from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.input_control.common.formatters.system_prompt import SystemPromptFormatter
from steerability.algorithms.input_control.common.memory.text import TextMemory
from steerability.algorithms.input_control.system_prompt.args import SystemPromptArgs


class SystemPrompt(InputControl):
    """Set or merge the leading system message of a chat.

    The text is a constant string configured at construction (no runtime kwargs). On chat input the leading
    system message (`chat[0]` when its role is `"system"`) is edited according to `mode`: `"prepend"` places the
    text ahead of the existing content, `"append"` after it, and `"replace"` substitutes it, joining with
    `separator` for `"prepend"` and `"append"`. When the chat has no leading system message, all three modes
    insert one carrying the text at position 0, so `separator` has no effect. Every input shape yields exactly
    one leading system message. System messages elsewhere in the chat are left untouched.

    The control implements both adaptation phases. The message phase (`adapt_messages`) is the faithful path for
    chat input, applied before chat templating. The token phase (`adapt`) is the fallback for raw text or tensor
    input; it decodes, re-templates, and re-encodes, and because decoding flattens any system prompt into text it
    cannot merge, so `mode` does not apply there and every mode behaves as `"replace"`. The base-class contract
    guarantees no double application: a non-`None` return from `adapt_messages` skips this control's `adapt` for
    that call.

    Args:
        text: System-prompt text set or merged into the leading system message. Non-empty.
        mode: How the text combines with an existing leading system message (`"prepend"` (default), `"append"`,
            or `"replace"`). Ignored when no leading system message is present.
        separator: String inserted between the text and the existing content for `"prepend"` and `"append"`.
            Empty string allowed.
    """

    Args = SystemPromptArgs

    supports_batching: bool = True

    # placeholders (dataclass attrs from SystemPromptArgs override these at __init__ time)
    tokenizer: PreTrainedTokenizerBase | None = None
    text: str | None = None
    mode: str = "prepend"
    separator: str = "\n\n"

    # method-owned state populated in steer()
    memory: TextMemory | None = None
    _formatter: SystemPromptFormatter | None = None

    def steer(
        self,
        model=None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.memory = TextMemory(slots={"instruction": self.text})
        self._formatter = SystemPromptFormatter(mode=self.mode, separator=self.separator)

    def adapt_messages(
        self,
        messages: list[list[dict]],
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]]:
        """Set or merge the leading system message of each chat.

        Always returns the adapted batch (never `None`), since `text` is validated non-empty and the control
        therefore always changes chat input.

        Args:
            messages: A batch of chats; outer list is the batch, inner list is one chat's message sequence.
            runtime_kwargs: Unused.

        Returns:
            The adapted batch of chats, each with exactly one leading system message.
        """
        return self._formatter.apply_to_messages(messages, self.memory)

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Set the system prompt on the token stream.

        Fallback path for raw text or tensor input; the message phase is the faithful path for chat input. The
        stream is decoded, re-templated with the text as the system message, and re-encoded, so `mode` does not
        apply and every mode behaves as `"replace"`. Handles `list[int]`, `list[list[int]]`, and 1-D or 2-D
        tensors, preserving the input container, dtype, and device on output. Batched sequences are padded to a
        uniform length.

        Args:
            input_ids: The user's prompt token ids.
            runtime_kwargs: Unused.

        Returns:
            The token ids with the system prompt applied.

        Raises:
            RuntimeError: If the tokenizer is not set (requires calling `steer()` first), or if a batch must be
                padded but the tokenizer has no `pad_token_id`.
        """
        if self.tokenizer is None:
            raise RuntimeError("SystemPrompt needs a tokenizer; call .steer() first.")

        is_tensor = isinstance(input_ids, torch.Tensor)
        original_device = input_ids.device if is_tensor else None
        original_dtype = input_ids.dtype if is_tensor else None

        # normalize to 2D list format [batch_size, seq_len]
        if is_tensor:
            if input_ids.ndim == 1:
                batch_input_ids = [input_ids.tolist()]
                single_sequence = True
            else:
                batch_input_ids = input_ids.tolist()
                single_sequence = False
        else:
            if isinstance(input_ids[0], int):
                batch_input_ids = [input_ids]
                single_sequence = True
            else:
                batch_input_ids = input_ids
                single_sequence = False

        adapted_batch: list[list[int]] = []
        for input_ids_single in batch_input_ids:
            input_tensor = torch.tensor(input_ids_single, dtype=torch.long).unsqueeze(0)
            adapted_tensor = self._formatter.apply_to_ids(input_tensor, self.memory, self.tokenizer)
            adapted_batch.append(adapted_tensor[0].tolist())

        max_len = max(len(seq) for seq in adapted_batch)
        if len(adapted_batch) > 1:
            if self.tokenizer.pad_token_id is None:
                raise RuntimeError(
                    "SystemPrompt: tokenizer has no pad_token_id; cannot pad batch sequences. "
                    "Set a pad token before using SystemPrompt with batched inputs."
                )
            pad_id = self.tokenizer.pad_token_id
            adapted_batch = [seq + [pad_id] * (max_len - len(seq)) for seq in adapted_batch]

        if is_tensor:
            result = torch.tensor(adapted_batch, dtype=original_dtype, device=original_device)
            if single_sequence:
                result = result.squeeze(0)
            return result
        if single_sequence:
            return adapted_batch[0]
        return adapted_batch
