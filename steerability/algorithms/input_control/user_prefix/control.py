"""
Input control that prepends a fixed text marker to a user turn.
"""
import torch
from transformers import PreTrainedTokenizerBase

from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.input_control.common.formatters.prepend_text import PrependTextFormatter
from steerability.algorithms.input_control.common.memory.text import TextMemory
from steerability.algorithms.input_control.user_prefix.args import UserPrefixArgs


class UserPrefix(InputControl):
    """Prepend a fixed text marker to a user turn.

    The marker is a constant string configured at construction (no runtime kwargs). On chat input the marker is
    concatenated to the content of the user turn(s) selected by `placement`, joined by `separator`; when a chat
    has no user turn a new user turn carrying the marker is appended. On raw text or tensor input the encoded
    marker is prefixed to the token stream.

    The control implements both adaptation phases. The message phase (`adapt_messages`) is the faithful path for
    chat input, applied before chat templating. The token phase (`adapt`) is the fallback for non-chat input and
    prefixes the encoded marker to the ids. The base-class contract guarantees no double application: a non-`None`
    return from `adapt_messages` skips this control's `adapt` for that call.

    Args:
        text: Marker text prepended to the targeted user turn(s). Non-empty.
        separator: String inserted between the marker and the existing user content. Empty string allowed.
        placement: Which user turn(s) receive the marker (`"last_user"` (default), `"first_user"`, or `"all_user"`).
    """

    Args = UserPrefixArgs

    supports_batching: bool = True

    # placeholders (dataclass attrs from UserPrefixArgs override these at __init__ time)
    tokenizer: PreTrainedTokenizerBase | None = None
    text: str | None = None
    separator: str = "\n\n"
    placement: str = "last_user"

    # method-owned state populated in steer()
    memory: TextMemory | None = None
    _formatter: PrependTextFormatter | None = None

    def steer(
        self,
        model=None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.memory = TextMemory(slots={"text": self.text})
        self._formatter = PrependTextFormatter(separator=self.separator, target=self.placement)

    def adapt_messages(
        self,
        messages: list[list[dict]],
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]]:
        """Prepend the marker to the targeted user turn of each chat.

        Always returns the adapted batch (never `None`), since `text` is validated non-empty and the control
        therefore always changes chat input.

        Args:
            messages: A batch of chats; outer list is the batch, inner list is one chat's message sequence.
            runtime_kwargs: Unused.

        Returns:
            The adapted batch of chats.
        """
        return self._formatter.apply_to_messages(messages, self.memory)

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Prefix the encoded marker to the token ids.

        Fallback path for raw text or tensor input; the message phase is the faithful path for chat input.
        Handles `list[int]`, `list[list[int]]`, and 1-D or 2-D tensors, preserving the input container, dtype, and
        device on output. Batched sequences are padded to a uniform length.

        Args:
            input_ids: The user's prompt token ids.
            runtime_kwargs: Unused.

        Returns:
            The token ids with the encoded marker prefixed.

        Raises:
            RuntimeError: If the tokenizer is not set (requires calling `steer()` first), or if a batch must be
                padded but the tokenizer has no `pad_token_id`.
        """
        if self.tokenizer is None:
            raise RuntimeError("UserPrefix needs a tokenizer; call .steer() first.")

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

        if len(adapted_batch) > 1:
            if self.tokenizer.pad_token_id is None:
                raise RuntimeError(
                    "UserPrefix: tokenizer has no pad_token_id; cannot pad batch sequences. "
                    "Set a pad token before using UserPrefix with batched inputs."
                )
            pad_id = self.tokenizer.pad_token_id
            max_len = max(len(seq) for seq in adapted_batch)
            adapted_batch = [seq + [pad_id] * (max_len - len(seq)) for seq in adapted_batch]

        if is_tensor:
            result = torch.tensor(adapted_batch, dtype=original_dtype, device=original_device)
            if single_sequence:
                result = result.squeeze(0)
            return result
        if single_sequence:
            return adapted_batch[0]
        return adapted_batch
