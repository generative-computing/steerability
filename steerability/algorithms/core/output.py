"""The `Output` generation record, per-row finish-reason inference, and the stop-string
truncation rule shared by every backend."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

FINISH_REASONS: tuple[str, ...] = ("stop", "eos", "length")


@dataclass(slots=True)
class Output:
    """The result of one generation call.

    Attributes:
        output_ids: Generated token IDs as a `[batch, seq]` tensor, excluding the prompt (the same
            slice the pipeline returns to the caller by default). Token ids are returned as
            generated; stop strings and any token-boundary overrun are not removed from them.
        adapted_input_ids: The `input_ids` actually fed to the model after all input-control
            transformations. For a padded batch these are in left-packed layout. None if not
            provided by the producer.
        finish_reason: The first row's finish reason, one of `"stop"`, `"eos"`, `"length"`, or
            None when none can be inferred.
        finish_reasons: Per-row finish reasons matching `output_ids` (one entry per candidate
            when the producer generated several), or None when the producer reports only the
            first row's reason.
        generated_tokens: The total tokens generated to produce this output, including rollouts a
            decoding driver proposed and discarded, or None when the producer did not count them.
            On the driver path the pipeline attaches the session wrapper's accumulated total; for a
            multi-row dispatch the total is split evenly across rows (integer division, remainder
            on the first row), so arm-level sums are preserved. The driverless path leaves it None,
            and consumers fall back to the non-pad count of `output_ids`.
    """
    output_ids: torch.Tensor
    adapted_input_ids: torch.Tensor | None = None
    finish_reason: str | None = None
    finish_reasons: tuple[str | None, ...] | None = None
    generated_tokens: int | None = None

    def decode(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode `output_ids` to text. Batch-aware."""
        return tokenizer.decode(
            self.output_ids, skip_special_tokens=skip_special_tokens
        )


def truncate_at_stop_strings(text: str, stop_strings: Sequence[str]) -> str:
    """Truncate `text` at the earliest occurrence of any stop string.

    This is the one client-side truncation rule applied to decoded continuation text on every
    backend. Token ids are never modified; only the decoded text is cut, at the start of the
    earliest match.

    With reasoning models the earliest-match rule interacts with thinking segments: a stop string
    that also occurs inside the reasoning (e.g. `"Answer:"`) cuts the text mid-thinking, so the
    thinking segment is left unclosed and a later split reports an empty answer. Choose stop strings
    that cannot appear before the closing think tag, or omit them when generating with thinking on.

    Args:
        text: Decoded continuation text.
        stop_strings: Stop strings; empty leaves `text` unchanged.

    Returns:
        `text` up to (excluding) the earliest stop-string occurrence, or `text` unchanged when
        no stop string occurs.
    """
    cut = len(text)
    for stop in stop_strings:
        if not stop:
            continue
        index = text.find(stop)
        if index != -1 and index < cut:
            cut = index
    return text[:cut]


def infer_finish_reasons(
    new_tokens: torch.Tensor,
    gen_kwargs: dict,
    *,
    eos_token_id: int | list[int] | None,
    pad_token_id: int | None,
    stop_strings: Sequence[str] = (),
    stop_token_ids: Sequence[int] = (),
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> list[str | None]:
    """Classify a per-row finish reason from generated token IDs and the composed stop rules.

    Args:
        new_tokens: Generated token IDs as a `[batch, gen_len]` tensor, right-padded by `generate`
            (the continuation only, with the prompt excluded).
        gen_kwargs: Generation parameters; only `max_new_tokens` is consulted.
        eos_token_id: End-of-sequence token ID(s); an int, a list of ints, or None. Normalized to a
            set of IDs internally.
        pad_token_id: Padding token ID used to right-pad short rows, or None.
        stop_strings: Stop strings composed for this generation; requires `tokenizer` to take
            effect.
        stop_token_ids: Extra stop token ids composed for this generation.
        tokenizer: Tokenizer used to decode continuations for the stop-string test, or None.

    Returns:
        One reason per row, in order, classified with the precedence stop, then eos, then
        length, then None. For row `i`, trailing `pad_token_id` positions are stripped to
        recover the true continuation length `n`, then:

            - `"stop"` if the decoded continuation contains a stop string (when a tokenizer is
              available), or the last unstripped token is one of `stop_token_ids`;
            - `"eos"` if `n > 0` and the last unstripped token is in the eos set, or
              `pad_token_id` is in the eos set and at least one trailing token was stripped (the
              pad-equals-eos configuration common to Llama-family tokenizers, where the first
              stripped token was the genuine EOS);
            - `"length"` if `max_new_tokens` is set and `n >= max_new_tokens`;
            - None otherwise (including zero-length rows).

    Stop rules the classifier was not given, such as caller-supplied custom stopping criteria,
    still classify as None.
    """
    eos_ids: set[int] = set()
    if isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    elif eos_token_id is not None:
        eos_ids = {int(token_id) for token_id in eos_token_id}

    stop_ids = {int(token_id) for token_id in stop_token_ids}
    stop_texts = [text for text in stop_strings if text]
    max_new = gen_kwargs.get("max_new_tokens")
    pad_equals_eos = pad_token_id is not None and pad_token_id in eos_ids

    reasons: list[str | None] = []
    for row in new_tokens:
        row_list = row.tolist()

        stripped_any = False
        if pad_token_id is not None:
            end = len(row_list)
            while end > 0 and row_list[end - 1] == pad_token_id:
                end -= 1
            stripped_any = end < len(row_list)
            row_list = row_list[:end]

        n = len(row_list)

        stopped = n > 0 and row_list[-1] in stop_ids
        if not stopped and stop_texts and tokenizer is not None and n > 0:
            continuation = tokenizer.decode(row_list, skip_special_tokens=False)
            stopped = any(text in continuation for text in stop_texts)

        if stopped:
            reasons.append("stop")
        elif n > 0 and row_list[-1] in eos_ids:
            reasons.append("eos")
        elif pad_equals_eos and stripped_any:
            reasons.append("eos")
        elif max_new is not None and n >= max_new:
            reasons.append("length")
        else:
            reasons.append(None)

    return reasons
