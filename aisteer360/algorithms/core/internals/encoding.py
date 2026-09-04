"""Text-to-tensor tokenization entry points."""
from typing import Sequence

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    device: torch.device | str | None = None,
    *,
    add_special_tokens: bool = True,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize a flat list of texts independently.

    Each text is tokenized on its own, without interleaving. Use this when positive and negative
    examples are independent and do not need co-padding for token alignment.

    Args:
        tokenizer: Tokenizer to use.
        texts: List of text strings.
        device: Target device for the returned tensors. When None, tensors are returned on the
            device the tokenizer produces them on (CPU).
        add_special_tokens: Whether to add special tokens (e.g. BOS). Pass False
            for chat-templated text that already contains them.
        max_length: Truncation bound. When None, truncation falls back to the tokenizer's model
            maximum length.

    Returns:
        Dictionary with input_ids and attention_mask tensors.
    """
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    if device is None:
        return dict(enc)
    return {k: v.to(device) for k, v in enc.items()}


def tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase,
    pos_texts: Sequence[str],
    neg_texts: Sequence[str],
    device: torch.device | str,
    *,
    add_special_tokens: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Tokenize positive/negative pairs together to ensure consistent padding.

    Interleaves pairs before tokenization so each (pos, neg) pair shares the same
    padding length, keeping shared prefixes token-aligned.

    Args:
        tokenizer: Tokenizer to use.
        pos_texts: List of positive text strings.
        neg_texts: List of negative text strings (same length as pos_texts).
        device: Target device.
        add_special_tokens: Whether to add special tokens (e.g. BOS). Pass False
            for chat-templated text that already contains them.

    Returns:
        Tuple of (enc_pos, enc_neg) dictionaries with input_ids and attention_mask.
    """
    # interleave: [pos0, neg0, pos1, neg1, ...]
    interleaved = []
    for pos, neg in zip(pos_texts, neg_texts):
        interleaved.append(pos)
        interleaved.append(neg)

    enc = tokenizer(
        interleaved,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # de-interleave: even indices are positive, odd indices are negative
    enc_pos = {k: v[0::2] for k, v in enc.items()}
    enc_neg = {k: v[1::2] for k, v in enc.items()}

    return enc_pos, enc_neg
