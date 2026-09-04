"""General-purpose, tokenizer-level rendering.
"""
from __future__ import annotations

import logging
from typing import Literal

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

PromptFormat = Literal["raw", "chat_completion", "chat_prompt"]


def has_chat_template(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Single source of truth for chat-template detection."""
    return getattr(tokenizer, "chat_template", None) is not None


def render_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render chat messages into a string via the tokenizer's chat template.

    Wrapper around `tokenizer.apply_chat_template(tokenize=False, ...)`.

    Args:
        tokenizer: Tokenizer whose chat template defines the rendering.
        messages: Chat messages as `{"role": ..., "content": ...}` dicts.
        add_generation_prompt: Whether to append the assistant generation prompt.

    Returns:
        The rendered string. When the tokenizer has no chat template, falls back
        (with a warning) to joining message contents with blank lines.

    Note:
        The returned string already contains the template's special tokens (e.g.
        BOS). Callers MUST tokenize it with `add_special_tokens=False` (see
        `encode_for_model`).
    """
    if not has_chat_template(tokenizer):
        logger.warning("render_messages: tokenizer has no chat_template; concatenating contents.")
        return "\n\n".join(m["content"] for m in messages)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def render_for_model(
    tokenizer: PreTrainedTokenizerBase,
    *,
    prompt: str | None = None,
    completion: str | None = None,
    mode: PromptFormat = "chat_completion",
) -> str:
    """Render a single example into the exact string the model will process.

    Args:
        tokenizer: Tokenizer whose chat template defines the rendering.
        prompt: The user-turn content (or, for `raw`, the left part of the
            concatenation).
        completion: The continuation appended after the generation prompt (or,
            for `raw`, the right part of the concatenation).
        mode: Rendering policy.

            - `"raw"`: concatenate `prompt + completion` verbatim (no template).
              For base-model methods (ActAdd) and standalone statements (ITI).
            - `"chat_completion"`: render `prompt` as a user turn (with generation
              prompt) and append `completion` as the continuation. The last
              content token is the completion's final token (correct for
              `accumulate="last_token"`). For prompt+answer contrastive pairs (CAA).
            - `"chat_prompt"`: render `prompt` as a user turn with generation
              prompt and no completion, producing the prompt as inference renders
              it. For standalone-prompt contrasts (CAST condition) and for the
              inference path itself.

    Returns:
        The rendered string. Falls back to `raw` (with a warning) when chat
        formatting is requested but the tokenizer has no chat template.

    Note:
        The returned string already contains the template's special tokens
        (e.g. BOS). Callers MUST tokenize it with `add_special_tokens=False`.
    """
    text = prompt or ""
    comp = completion or ""

    if mode == "raw":
        return text + comp

    if not has_chat_template(tokenizer):
        logger.warning(
            "render_for_model: mode=%r requested but tokenizer has no chat_template; "
            "falling back to raw concatenation.",
            mode,
        )
        return text + comp

    head = render_messages(tokenizer, [{"role": "user", "content": text}], add_generation_prompt=True)
    return head if mode == "chat_prompt" else head + comp


def encode_for_model(
    tokenizer: PreTrainedTokenizerBase,
    *,
    prompt: str | None = None,
    completion: str | None = None,
    messages: list[dict[str, str]] | None = None,
    mode: PromptFormat = "chat_prompt",
    **tokenizer_kwargs,
):
    """Render then tokenize with the correct `add_special_tokens`.

    Convenience for single-prompt call sites (e.g. judges).

    A template is applied iff `has_chat_template(tokenizer)` and
    (for the `prompt` path) `mode != "raw"`; in that case the
    rendered string already contains the special tokens, so it is
    tokenized with `add_special_tokens=False`, otherwise `True`.

    Args:
        tokenizer: Tokenizer to render and tokenize with.
        prompt: User-turn content for the `render_for_model` path.
        completion: Continuation for the `render_for_model` path.
        messages: Chat messages for the `render_messages` path. Takes precedence
            over `prompt`/`completion` when provided.
        mode: Rendering policy for the `render_for_model` path (ignored when
            `messages` is given).
        **tokenizer_kwargs: Forwarded to the tokenizer call (e.g. `return_tensors`).

    Returns:
        The tokenizer output (a `BatchEncoding`).
    """
    if messages is not None:
        text = render_messages(tokenizer, messages)
        template_applied = has_chat_template(tokenizer)
    else:
        text = render_for_model(tokenizer, prompt=prompt, completion=completion, mode=mode)
        template_applied = has_chat_template(tokenizer) and mode != "raw"
    return tokenizer(text, add_special_tokens=not template_applied, **tokenizer_kwargs)
