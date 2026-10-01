"""The `Output` generation record, per-row finish-reason inference, and the stop-string
truncation rule shared by every backend."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

FINISH_REASONS: tuple[str, ...] = ("stop", "eos", "length")


@dataclass(slots=True, kw_only=True)
class Output:
    """The result of one generation call.

    The constructor checks that `output_ids` is two-dimensional, that `finish_reasons` has one
    entry per row of `output_ids`, and that every entry is one of `FINISH_REASONS` or None. A
    single-row output reads its reason as `finish_reasons[0]`.

    Attributes:
        output_ids: Generated token IDs as a `[batch, seq]` tensor, excluding the prompt (the same
            slice the pipeline returns to the caller by default). Token ids are returned as
            generated; stop strings and any token-boundary overrun are not removed from them.
        adapted_input_ids: The `input_ids` actually fed to the model after all input-control
            transformations. For a padded batch these are in left-packed layout. None if not
            provided by the producer.
        finish_reasons: One reason per row of `output_ids`, in row order (one per candidate when
            the producer generated several). Each entry is `"stop"`, `"eos"`, `"length"`, or None
            when no reason could be inferred for that row.
        generated_tokens: The total tokens generated to produce this output, including rollouts a
            decoding driver proposed and discarded, or None when the producer did not count them.
            On the driver path the pipeline attaches the session wrapper's accumulated total; for a
            multi-row dispatch the total is split evenly across rows (integer division, remainder
            on the first row), so arm-level sums are preserved. The driverless path leaves it None,
            and consumers fall back to the non-pad count of `output_ids`.
    """
    output_ids: torch.Tensor
    adapted_input_ids: torch.Tensor | None = None
    finish_reasons: tuple[str | None, ...]
    generated_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.output_ids.dim() != 2:
            raise ValueError(f"output_ids must be [batch, seq]; got shape {tuple(self.output_ids.shape)}.")
        if len(self.finish_reasons) != self.output_ids.size(0):
            raise ValueError(
                f"finish_reasons has {len(self.finish_reasons)} entries for {self.output_ids.size(0)} rows of "
                "output_ids."
            )
        for reason in self.finish_reasons:
            if reason is not None and reason not in FINISH_REASONS:
                raise ValueError(f"finish_reasons entries are one of {FINISH_REASONS} or None; got {reason!r}.")

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
    """Infer each row's finish reason from its generated token ids and the composed stop rules.

    Trailing `pad_token_id` positions are removed from each row to recover its continuation length
    `n`. The first rule that matches gives the row's reason:

    1. `"stop"` if `n > 0` and the last token is one of `stop_token_ids`, or if the decoded
       continuation contains a stop string (only when `tokenizer` is given).
    2. `"eos"` if `n > 0` and the last token is an eos id.
    3. `"length"` if `max_new_tokens` is set and `n >= max_new_tokens`.
    4. `"eos"` if `pad_token_id` is an eos id and at least one trailing pad was removed. In this
       configuration (common to Llama-family tokenizers), the first removed pad is the eos the model
       emitted.
    5. None otherwise.

    The length rule comes before the pad-equals-eos rule because a row with `max_new_tokens` tokens
    reached the limit, and its trailing pads only extend it to the length of a longer row in the batch
    (a decoding driver can return rows longer than `max_new_tokens`). A row that stopped on a rule the
    function is not given, such as a custom stopping criterion from the caller, gets None unless
    another rule matches.

    Args:
        new_tokens: Generated token ids of shape `[batch, gen_len]`, right-padded, with the prompt
            excluded.
        gen_kwargs: Generation parameters. Only `max_new_tokens` is read.
        eos_token_id: The eos token id, a list of eos token ids, or None.
        pad_token_id: The pad token id used to right-pad short rows, or None.
        stop_strings: Stop strings composed for this generation. They take effect only when
            `tokenizer` is given.
        stop_token_ids: Stop token ids composed for this generation.
        tokenizer: Tokenizer that decodes continuations for the stop-string test, or None.

    Returns:
        One finish reason per row, in row order: `"stop"`, `"eos"`, `"length"`, or None.
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
        elif max_new is not None and n >= max_new:
            reasons.append("length")
        elif pad_equals_eos and stripped_any:
            reasons.append("eos")
        else:
            reasons.append(None)

    return reasons
