"""Shared batched generation helper for single-system-prompt input controls.

`generate_with_system_prompt` runs the task LM over a batch of queries under one system prompt. It applies
the tokenizer's chat template when available (otherwise a plain `system\\n\\nquery` join), left-pads for the
causal-LM batch, restores the tokenizer's padding side, and decodes only the continuation.
"""
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def generate_with_system_prompt(
    task_lm,
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    queries: list[str],
    gen_kwargs: dict | None = None,
) -> list[str]:
    """Generate one continuation per query under a shared system prompt.

    Args:
        task_lm: Causal language model.
        tokenizer: Tokenizer paired with `task_lm`.
        system_prompt: System prompt prepended to every query.
        queries: User-facing query texts (one continuation generated per query).
        gen_kwargs: Forwarded to `task_lm.generate`. Defaults to greedy, `max_new_tokens=32`.

    Returns:
        Decoded continuations, one per query, in order.
    """
    if not queries:
        return []
    gen_kwargs = dict(gen_kwargs or {"max_new_tokens": 32, "do_sample": False})

    used_template = bool(getattr(tokenizer, "chat_template", None))
    if used_template:
        prompts_text = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for query in queries
        ]
    else:
        prompts_text = [f"{system_prompt}\n\n{query}" for query in queries]

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=not used_template,
        ).to(task_lm.device)

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is not None:
            gen_kwargs.setdefault("pad_token_id", pad_id)

        with torch.no_grad():
            output_ids = task_lm.generate(**encoded, **gen_kwargs)

        prompt_len = encoded["input_ids"].size(1)
        new_ids = output_ids[:, prompt_len:]
        decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = original_padding_side

    return decoded
