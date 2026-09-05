"""Render `ContrastivePairs` into model-ready text.

The general, tokenizer-level rendering primitives live in
`steerability.utils.rendering`; this module adds the data-specific renderer
that turns a `ContrastivePairs` into model-ready text under a `PromptFormat`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from steerability.algorithms.core.internals.data import ContrastivePairs, LabeledExamples
from steerability.utils.rendering import PromptFormat, has_chat_template, render_for_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderedContrastive:
    """Rendered text for a `ContrastivePairs`, plus tokenization policy.

    Attributes:
        pos_texts: Rendered positive examples.
        neg_texts: Rendered negative examples.
        prompt_texts: User-turn-only renders for suffix-only span computation, or
            None when there is no separate prompt/completion split.
        add_special_tokens: Flag callers must pass to the tokenizer for all of
            the above (False for chat modes, True for raw).
        effective_mode: The mode actually applied after raw fallbacks.
    """

    pos_texts: list[str]
    neg_texts: list[str]
    prompt_texts: list[str] | None
    add_special_tokens: bool
    effective_mode: PromptFormat


def render_contrastive(
    tokenizer: PreTrainedTokenizerBase,
    data: ContrastivePairs | LabeledExamples,
    mode: PromptFormat,
) -> RenderedContrastive:
    """Render both sides of a contrastive dataset under `mode`.

    Accepts `ContrastivePairs` (paired, with optional shared prompts) or
    `LabeledExamples` (unpaired classes, rendered independently). Resolves the
    effective mode (with raw fallbacks and warnings), renders
    positives/negatives, renders the prompt-only strings used for suffix-only
    span computation, and reports the `add_special_tokens` flag the tokenizer
    must use for all of the above.

    Args:
        tokenizer: Tokenizer whose chat template defines the rendering.
        data: `ContrastivePairs` (with `positives`, `negatives`, and optional
            `prompts`) or `LabeledExamples`.
        mode: Requested rendering policy.

    Returns:
        A `RenderedContrastive` with rendered texts and tokenization policy.

    Raises:
        ValueError: If `mode` is `"chat_completion"` and `data` is
            `LabeledExamples` (that format's shared prompts align per example,
            which requires paired data).
    """
    if mode == "chat_completion" and isinstance(data, LabeledExamples):
        raise ValueError(
            "prompt_format='chat_completion' requires paired data with shared prompts "
            "(ContrastivePairs); LabeledExamples classes are unpaired."
        )
    prompts = getattr(data, "prompts", None)

    effective: PromptFormat = mode

    if mode in ("chat_completion", "chat_prompt") and not has_chat_template(tokenizer):
        logger.warning("render_contrastive: no chat_template; falling back to raw.")
        effective = "raw"

    if mode == "chat_completion" and prompts is None:
        logger.warning(
            "prompt_format='chat_completion' requires `prompts` (positives/negatives "
            "are treated as completions); none provided. Falling back to raw."
        )
        effective = "raw"

    add_special = effective == "raw"

    if effective == "raw":
        if prompts is not None:
            pos = [p + c for p, c in zip(prompts, data.positives)]
            neg = [p + c for p, c in zip(prompts, data.negatives)]
            prompt_texts = list(prompts)
        else:
            pos = list(data.positives)
            neg = list(data.negatives)
            prompt_texts = None
    elif effective == "chat_completion":
        pos = [
            render_for_model(tokenizer, prompt=p, completion=c, mode="chat_completion")
            for p, c in zip(prompts, data.positives)
        ]
        neg = [
            render_for_model(tokenizer, prompt=p, completion=c, mode="chat_completion")
            for p, c in zip(prompts, data.negatives)
        ]
        prompt_texts = [render_for_model(tokenizer, prompt=p, mode="chat_prompt") for p in prompts]
    else:  # chat_prompt mode, where each positive/negative is a standalone prompt
        pos = [render_for_model(tokenizer, prompt=t, mode="chat_prompt") for t in data.positives]
        neg = [render_for_model(tokenizer, prompt=t, mode="chat_prompt") for t in data.negatives]
        prompt_texts = None  # no separate prompt/completion split

    return RenderedContrastive(pos, neg, prompt_texts, add_special, effective)
