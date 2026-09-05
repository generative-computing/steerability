"""Tokenizer- and attention-mask-level hygiene: pad-token setup, token counting, mask inference, batch re-padding,
and BOS sanity checks."""
from __future__ import annotations

import logging

import torch
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Set pad token to eos token if not already defined.

    Args:
       tokenizer: HuggingFace tokenizer instance

    Returns:
       The same tokenizer with pad_token configured
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Count the tokens in `text`, without special tokens.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        text: The text to tokenize.

    Returns:
        The token count.
    """
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def infer_attention_mask_from_ids(
        input_ids: torch.Tensor,
        pad_token_id: int | None,
) -> torch.Tensor:
    """Best-effort attention mask when the caller supplied none.

    Marks only leading and trailing runs of `pad_token_id` as padding; interior occurrences are kept
    as real tokens. This matters when pad == eos (set by `ensure_pad_token` for tokenizers without a
    pad token): chat templates legitimately contain the eos mid-prompt (e.g. ChatML `<|im_end|>`), and
    a token-identity mask (`ids != pad_id`) would wrongly mask those positions.

    Rules:

    - `pad_token_id` is None -> all ones.
    - a row that is entirely pad -> all ones (degenerate; safest).
    - otherwise ones on `[first_nonpad, last_nonpad]`, zeros outside.

    Args:
        input_ids: Token ids of shape [batch, seq_len].
        pad_token_id: The tokenizer's pad token id, or None.

    Returns:
        A long tensor of shape [batch, seq_len] on `input_ids.device`.
    """
    if pad_token_id is None:
        return torch.ones_like(input_ids, dtype=torch.long)

    _, seq_len = input_ids.shape
    nonpad = input_ids != pad_token_id
    positions = torch.arange(seq_len, device=input_ids.device)
    first = torch.where(nonpad, positions, seq_len).min(dim=1).values  # [batch]
    last = torch.where(nonpad, positions, -1).max(dim=1).values  # [batch]
    mask = (positions.unsqueeze(0) >= first.unsqueeze(1)) & (positions.unsqueeze(0) <= last.unsqueeze(1))
    mask[first == seq_len] = True  # all-pad rows -> all ones
    return mask.long()


def to_left_pad(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rearrange a batch to left-padded layout for correct batched causal scoring.

    Left-padding ensures that all sequences in the batch end at the same position, allowing a uniform logit slice
    after concatenation with reference tokens. Works correctly regardless of whether the input is right-padded,
    left-padded, or unpadded.

    Args:
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Corresponding attention mask [batch, seq_len]

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Left-padded (input_ids, attention_mask)
    """
    batch_size, max_len = input_ids.shape
    seq_lens = attention_mask.sum(dim=1)

    left_ids = input_ids.clone()
    left_mask = torch.zeros_like(attention_mask)

    for i in range(batch_size):
        length = seq_lens[i]
        pad = max_len - length
        if pad > 0:
            real_tokens = input_ids[i][attention_mask[i].bool()]
            pad_tokens = input_ids[i][~attention_mask[i].bool()]
            left_ids[i] = torch.cat([pad_tokens, real_tokens])
        left_mask[i, max_len - length:] = 1

    return left_ids, left_mask


def warn_if_duplicate_bos(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase | None,
        already_warned: bool,
) -> bool:
    """Warn once if any row starts with two BOS tokens; returns the updated warned-state.

    The dominant cause is rendering a chat template to a string and re-tokenizing it with
    `add_special_tokens=True`; the template already contains the BOS.

    Returns `already_warned` unchanged (no re-check) when `already_warned` is True, when `tokenizer`
    is None, when `tokenizer.bos_token_id` is None, or when seq_len < 2. Returns True after emitting
    the warning.

    Args:
        input_ids: The (2D) prompt token ids about to be generated from.
        attention_mask: The matching attention mask (used to locate the first real token, so the
            check is left-padding safe).
        tokenizer: The pipeline tokenizer (or None).
        already_warned: Whether the duplicate-BOS warning has already fired for this pipeline.

    Returns:
        The updated warned-state.
    """
    if already_warned or tokenizer is None:
        return already_warned
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None or input_ids.size(1) < 2:
        return already_warned
    first = attention_mask.long().argmax(dim=1)  # first real token (left-pad safe)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    second = (first + 1).clamp(max=input_ids.size(1) - 1)
    dup = (
        (input_ids[rows, first] == bos)
        & (input_ids[rows, second] == bos)
        & (first + 1 < input_ids.size(1))
    )
    if bool(dup.any()):
        logger.warning(
            "Duplicate BOS detected at the start of the prompt (token id %d twice). "
            "Likely cause: chat-templated text re-tokenized with add_special_tokens=True. "
            "Tokenize with add_special_tokens=False, use "
            "steerability.utils.rendering.encode_for_model, or pass chat messages "
            "directly to generate(). Steering methods calibrated on single-BOS "
            "inputs will misbehave on double-BOS inputs.", bos,
        )
        return True
    return already_warned
