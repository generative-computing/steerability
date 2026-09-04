"""Prompt resolution helpers for `SteeringPipeline`: source dispatch, per-modality validation
and tokenization, message-level adaptation, and the token-level adapt chain."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from aisteer360.algorithms.core.utils.controls import warn_if_adapt_messages_bypassed
from aisteer360.utils.tokenization import infer_attention_mask_from_ids, warn_if_duplicate_bos

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from aisteer360.algorithms.input_control.base import InputControl


@dataclass
class PromptWarnings:
    """Once-per-pipeline warning flags for prompt resolution."""

    tensor_with_adapt_messages: bool = False
    duplicate_bos: bool = False


def apply_adapt_messages_and_tokenize(
        input_controls: "list[InputControl]",
        tokenizer: "PreTrainedTokenizerBase",
        messages_batch: list[list[dict]],
        runtime_kwargs: dict,
        chat_template_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, set[int]]:
    """Fold every input control's `adapt_messages` over the message batch, then chat-template tokenize once.

    Controls run in list order. A non-None return becomes the input to the next control and marks
    that control as handled at message level; a None return passes the messages through unchanged
    and leaves the control unmarked, so the pipeline later runs its token-level `adapt` instead.
    Each control is therefore applied exactly once per call.

    Args:
        input_controls: Input controls whose `adapt_messages` runs in list order.
        tokenizer: Tokenizer whose `apply_chat_template` performs the tokenization.
        messages_batch: One conversation per row, each a list of chat-message mappings.
        runtime_kwargs: Per-call parameters forwarded to `adapt_messages`.
        chat_template_kwargs: Extra keyword arguments forwarded to `apply_chat_template` after the
            four pipeline-owned kwargs (`return_tensors`, `padding`, `add_generation_prompt`,
            `return_dict`). None or an empty mapping adds nothing. The toolkit does not interpret
            the keys; they are model-family specific (e.g. `enable_thinking`).

    Returns:
        tuple[input_ids, attention_mask, handled] where `handled` contains `id(control)` for each
        control whose `adapt_messages` returned a non-None result.
    """
    handled: set[int] = set()
    for control in input_controls:
        adapted = control.adapt_messages(
            messages_batch,
            runtime_kwargs=runtime_kwargs,
        )
        if adapted is not None:
            messages_batch = adapted
            handled.add(id(control))

    encoded = tokenizer.apply_chat_template(
        messages_batch,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        **(chat_template_kwargs or {}),
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask, handled


def resolve_generate_source(
        inputs: Any,
        text: Any,
        messages: Any,
        input_ids: Any,
) -> tuple[Literal["text", "messages", "tokens"], Any]:
    """Select the single prompt source and its modality.

    Exactly one of positional `inputs`, `text=`, `messages=`, or `input_ids=` may be provided.
    Positional input is a convenience for text prompts (`str` or a `list` whose every element is
    a `str`) and routes to text; any other positional shape raises (E12). Because the check is a
    total `all(...)` over the list, a mixed list such as `["a", {"role": ...}]` fails here rather
    than downstream.

    Returns:
        tuple[kind, payload] where `kind` is `"text"`, `"messages"`, or `"tokens"` and `payload`
        is the value handed to the matching resolver.

    Raises:
        TypeError: If no source or more than one source is provided (E1/E2), or a positional
            input is neither a `str` nor a `list[str]` (E12).
    """
    provided = [
        name for name, value in (
            ("inputs", inputs), ("text", text), ("messages", messages), ("input_ids", input_ids),
        ) if value is not None
    ]
    if len(provided) == 0:
        raise TypeError(
            "generate() requires a prompt: pass positional text, or exactly one of text=, "
            "messages=, input_ids=."
        )
    if len(provided) > 1:
        names = ", ".join(provided)
        raise TypeError(
            f"generate() received multiple prompt sources ({names}); pass exactly one of "
            "positional inputs, text=, messages=, input_ids=."
        )

    if text is not None:
        return "text", text
    if messages is not None:
        return "messages", messages
    if input_ids is not None:
        return "tokens", input_ids

    # positional inputs: text convenience only
    if isinstance(inputs, str) or (
        isinstance(inputs, list) and all(isinstance(element, str) for element in inputs)
    ):
        return "text", inputs
    raise TypeError(
        "positional input to generate() must be a str or list of str; pass messages=... "
        "for chat or input_ids=... for token input."
    )


def resolve_text_prompt(
        text: Any,
        *,
        input_controls: "list[InputControl]",
        tokenizer: "PreTrainedTokenizerBase",
        warnings_state: PromptWarnings,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """Validate and tokenize a text prompt (design §4.3.1).

    Args:
        text: A `str` (single) or a `list`/`tuple` whose elements are all `str` (batch).
        input_controls: Input controls consulted for the bypass warning.
        tokenizer: Tokenizer used for plain-text tokenization.
        warnings_state: Once-per-pipeline warning flags, updated in place.

    Returns:
        tuple[input_ids, attention_mask, is_single].

    Raises:
        TypeError: If `text` is a sequence containing a non-`str` element (E3).
        ValueError: If `text` is an empty sequence (E4).
    """
    is_single = isinstance(text, str)
    if is_single:
        normalized = [text]
    else:
        normalized = list(text)
        if len(normalized) == 0:
            raise ValueError("text= received an empty sequence.")
        for index, element in enumerate(normalized):
            if not isinstance(element, str):
                raise TypeError(
                    f"text= must be a str or a sequence of str; element {index} is "
                    f"{type(element).__name__}."
                )

    warnings_state.tensor_with_adapt_messages = warn_if_adapt_messages_bypassed(
        input_controls, warnings_state.tensor_with_adapt_messages
    )
    tokenized = tokenizer(normalized, return_tensors="pt", padding=True)
    return tokenized["input_ids"], tokenized.get("attention_mask"), is_single


def resolve_messages_prompt(
        messages: Any,
        runtime_kwargs: dict,
        *,
        input_controls: "list[InputControl]",
        tokenizer: "PreTrainedTokenizerBase",
        chat_template_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, set[int], bool]:
    """Validate a chat prompt, then adapt and chat-template tokenize it (design §4.3.2).

    Accepts one conversation (a sequence of mappings) or a batch (a sequence of sequences of
    mappings). Message elements are validated as `collections.abc.Mapping`; role/content schema
    remains the responsibility of `apply_chat_template`.

    Args:
        messages: One conversation or a batch of conversations.
        runtime_kwargs: Per-call parameters forwarded to `adapt_messages`.
        input_controls: Input controls whose `adapt_messages` runs in list order.
        tokenizer: Tokenizer whose `apply_chat_template` performs the tokenization.
        chat_template_kwargs: Extra keyword arguments forwarded to `apply_chat_template` after the
            pipeline-owned kwargs. None or an empty mapping adds nothing.

    Returns:
        tuple[input_ids, attention_mask, message_handled, is_single], where `message_handled`
        holds `id()`s of controls that adapted at message level.

    Raises:
        ValueError: If the conversation or batch is empty (E5).
        TypeError: If a batch inner element is not a mapping (E6) or the outer sequence mixes
            element kinds (E7).
    """
    outer = list(messages)
    if len(outer) == 0:
        raise ValueError("messages= received an empty conversation or batch.")

    if all(isinstance(element, Mapping) for element in outer):
        is_single = True
        normalized = [list(outer)]
    elif all(isinstance(element, (list, tuple)) for element in outer):
        is_single = False
        normalized = []
        for i, chat in enumerate(outer):
            chat = list(chat)
            if len(chat) == 0:
                raise ValueError("messages= received an empty conversation or batch.")
            for j, message in enumerate(chat):
                if not isinstance(message, Mapping):
                    raise TypeError(
                        f"messages[{i}][{j}] must be a mapping (one chat message); got "
                        f"{type(message).__name__}."
                    )
            normalized.append(chat)
    else:
        raise TypeError(
            "messages= must be one conversation (a sequence of mappings) or a batch (a sequence "
            "of sequences of mappings); got mixed element types at the outer level."
        )

    input_ids, attention_mask, message_handled = apply_adapt_messages_and_tokenize(
        input_controls, tokenizer, normalized, runtime_kwargs,
        chat_template_kwargs=chat_template_kwargs,
    )
    return input_ids, attention_mask, message_handled, is_single


def resolve_token_prompt(
        input_ids: Any,
        attention_mask: torch.Tensor | None,
        *,
        input_controls: "list[InputControl]",
        warnings_state: PromptWarnings,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """Validate a token prompt (tokens only; design §4.3.3).

    Args:
        input_ids: A 1-D/2-D `torch.Tensor`, a `list[int]`, or a `list[list[int]]`.
        attention_mask: Optional mask, passed through unchanged.
        input_controls: Input controls consulted for the bypass warning.
        warnings_state: Once-per-pipeline warning flags, updated in place.

    Returns:
        tuple[input_ids, attention_mask, is_single].

    Raises:
        ValueError: If a tensor is neither 1-D nor 2-D (E8), or nested lists are ragged (E9).
        TypeError: If the value is not a token tensor or integer list (E10).
    """
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 1:
            resolved, is_single = input_ids.unsqueeze(0), True
        elif input_ids.ndim == 2:
            resolved, is_single = input_ids, False
        else:
            raise ValueError(f"input_ids tensor must be 1-D or 2-D; got {input_ids.ndim}-D.")
    elif isinstance(input_ids, list) and input_ids and all(isinstance(x, int) for x in input_ids):
        resolved, is_single = torch.tensor([input_ids], dtype=torch.long), True
    elif (
        isinstance(input_ids, list) and input_ids
        and all(isinstance(row, list) and row and all(isinstance(x, int) for x in row) for row in input_ids)
    ):
        try:
            resolved = torch.tensor(input_ids, dtype=torch.long)
        except ValueError as exception:
            raise ValueError(
                "input_ids= nested lists must be rectangular (equal-length rows)."
            ) from exception
        is_single = False
    else:
        raise TypeError(
            f"input_ids= accepts a 1-D/2-D integer tensor, list[int], or list[list[int]]; got "
            f"{type(input_ids).__name__}. For text prompts use text= or positional input; for "
            "chat use messages=."
        )

    warnings_state.tensor_with_adapt_messages = warn_if_adapt_messages_bypassed(
        input_controls, warnings_state.tensor_with_adapt_messages
    )
    return resolved, attention_mask, is_single


def prepare_inputs(
        input_ids: list[int] | torch.LongTensor,
        attention_mask: torch.Tensor | None,
        *,
        input_controls: "list[InputControl]",
        tokenizer: "PreTrainedTokenizerBase | None",
        device: torch.device,
        runtime_kwargs: dict | None,
        message_handled: frozenset[int] = frozenset(),
        warnings_state: PromptWarnings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the token-level input-control chain and normalize input tensors.

    Runs each input control's `adapt` in list order (each control receives the previous
    control's output), then ensures both input_ids and attention_mask are properly shaped
    tensors on the correct device.

    Args:
        input_ids: Input token IDs as list or tensor [seq_len] or [batch, seq_len]
        attention_mask: Optional attention mask matching input_ids shape
        input_controls: Input controls whose token-level `adapt` runs in list order.
        tokenizer: Tokenizer consulted for pad-token mask inference and the duplicate-bos check.
        device: Device the returned tensors are moved to.
        runtime_kwargs: Per-call parameters for input controls
        message_handled: `id()`s of input controls whose `adapt_messages` already performed the
            adaptation before tokenization for this call; their token-level `adapt` is skipped so
            no control is applied twice to the same prompt.
        warnings_state: Once-per-pipeline warning flags, updated in place.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (steered_input_ids, attention_mask), both as 2D tensors on `device`
    """
    runtime_kwargs = runtime_kwargs or {}

    # token-phase chain (controls already handled at message level are skipped)
    steered_input_ids = input_ids
    for control in input_controls:
        if id(control) in message_handled:
            continue
        steered_input_ids = control.adapt(
            steered_input_ids,
            runtime_kwargs=runtime_kwargs,
        )

    # normalize input_ids to 2D tensor
    if isinstance(steered_input_ids, list):
        steered_input_ids = torch.tensor(steered_input_ids, dtype=torch.long)
    if steered_input_ids.ndim == 1:
        steered_input_ids = steered_input_ids.unsqueeze(0)
    steered_input_ids = steered_input_ids.to(device)

    # normalize attention_mask
    if attention_mask is not None:
        if isinstance(attention_mask, list):
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        # rebuild if length mismatch after input control transformation
        if attention_mask.shape[-1] != steered_input_ids.shape[-1]:
            attention_mask = None

    if attention_mask is None:
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            attention_mask = infer_attention_mask_from_ids(steered_input_ids, tokenizer.pad_token_id)
        else:
            attention_mask = torch.ones_like(steered_input_ids, dtype=torch.long)

    attention_mask = attention_mask.to(dtype=steered_input_ids.dtype, device=device)

    warnings_state.duplicate_bos = warn_if_duplicate_bos(
        steered_input_ids, attention_mask, tokenizer, warnings_state.duplicate_bos
    )

    return steered_input_ids, attention_mask
